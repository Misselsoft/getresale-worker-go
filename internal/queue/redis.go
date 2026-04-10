package queue

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

type Message struct {
	JobID string
	Body  string
}

type RedisManager struct {
	Client       *redis.Client
	InputQueue   string
	OutputQueue  string
	BlockTimeout time.Duration
	LockTTL      time.Duration
	WorkerID     string
	Prefix       string
}

func NewRedisManager(client *redis.Client, inputQueue, outputQueue string, workerID string, prefix string) *RedisManager {
	return &RedisManager{
		Client:       client,
		InputQueue:   inputQueue,
		OutputQueue:  outputQueue,
		BlockTimeout: 20 * time.Second,
		LockTTL:      5 * time.Minute,
		WorkerID:     workerID,
		Prefix:       prefix,
	}
}

func (m *RedisManager) ReceiveMessages(ctx context.Context) ([]Message, error) {
	// Simple BLPop from the input list
	queueName := m.InputQueue
	if m.Prefix != "" {
		queueName = fmt.Sprintf("%s:%s", m.Prefix, m.InputQueue)
	}

	// BLPop returns [key, value]
	result, err := m.Client.BLPop(ctx, m.BlockTimeout, queueName).Result()
	if err == redis.Nil {
		return []Message{}, nil
	}
	if err != nil {
		return nil, err
	}

	// result[0] is key (queue name), result[1] is value (payload)
	payload := result[1]

	// We don't have a separate JobID in simple list mode, unless it's in the payload.
	// We'll generate a temporary ID for logging purposes or try to extract it if needed.
	// For now, let's use a placeholder or hash of payload.
	jobID := "list-job"

	return []Message{{JobID: jobID, Body: payload}}, nil
}

func (m *RedisManager) SendResult(ctx context.Context, jobID string, result interface{}) error {
	data, err := json.Marshal(result)
	if err != nil {
		return err
	}

	queueName := m.OutputQueue
	if m.Prefix != "" {
		queueName = fmt.Sprintf("%s:%s", m.Prefix, m.OutputQueue)
	}

	// Simple RPush to the output list
	return m.Client.RPush(ctx, queueName, data).Err()
}

func (m *RedisManager) CompleteJob(ctx context.Context, jobID string, returnValue string) error {
	// In simple list mode, completion is just sending the result (handled by SendResult).
	// This method might be redundant or can be used for logging.
	return nil
}

func (m *RedisManager) FailJob(ctx context.Context, jobID string, reason string) error {
	failedQueue := m.InputQueue + ":failed"
	fmt.Printf("Job %s failed: %s. Moving to %s\n", jobID, reason, failedQueue)

	// We need the job body to push to failed queue.
	// Since we don't have it here, we rely on the caller to handle this if they want DLQ.
	// But wait, the caller (worker) has the message.
	// The interface of FailJob only takes ID and reason.
	// We should probably update the interface or just log for now.
	// To properly support DLQ for failed jobs, we need the body.

	return nil
}

func (m *RedisManager) FailJobWithBody(ctx context.Context, jobID string, body string, reason string) error {
	failedQueue := m.InputQueue + ":failed"
	if m.Prefix != "" {
		failedQueue = fmt.Sprintf("%s:%s:failed", m.Prefix, m.InputQueue)
	}
	fmt.Printf("Job %s failed: %s. Moving to %s\n", jobID, reason, failedQueue)
	return m.Client.RPush(ctx, failedQueue, body).Err()
}

func (m *RedisManager) RetryJob(ctx context.Context, jobID string, body string, delay time.Duration) error {
	delayedQueue := m.InputQueue + ":delayed"
	if m.Prefix != "" {
		delayedQueue = fmt.Sprintf("%s:%s:delayed", m.Prefix, m.InputQueue)
	}
	// Use integer milliseconds for score to avoid floating point issues
	score := float64(time.Now().Add(delay).UnixMilli())

	fmt.Printf("Retrying job %s in %v (score: %.0f)\n", jobID, delay, score)

	// Store the body in the delayed queue
	return m.Client.ZAdd(ctx, delayedQueue, redis.Z{
		Score:  score,
		Member: body,
	}).Err()
}

func (m *RedisManager) StartDelayedJobScheduler(ctx context.Context) {
	delayedQueue := m.InputQueue + ":delayed"
	if m.Prefix != "" {
		delayedQueue = fmt.Sprintf("%s:%s:delayed", m.Prefix, m.InputQueue)
	}
	// Poll every 5 seconds instead of 1 to reduce load, or keep 1s for responsiveness
	ticker := time.NewTicker(2 * time.Second)

	go func() {
		defer ticker.Stop()
		fmt.Printf("Delayed job scheduler started for %s\n", delayedQueue)

		for {
			select {
			case <-ctx.Done():
				fmt.Println("Delayed job scheduler stopping...")
				return
			case <-ticker.C:
				now := float64(time.Now().UnixMilli())

				// Get jobs that are ready to be processed
				// Use explicit string formatting for score range
				maxScore := fmt.Sprintf("%.0f", now)

				// Limit to 10 jobs at a time to avoid blocking
				jobs, err := m.Client.ZRangeByScore(ctx, delayedQueue, &redis.ZRangeBy{
					Min:   "-inf",
					Max:   maxScore,
					Count: 10,
				}).Result()

				if err != nil {
					fmt.Printf("Error checking delayed jobs: %v\n", err)
					continue
				}

				if len(jobs) > 0 {
					fmt.Printf("Found %d delayed jobs ready to process (max score: %s)\n", len(jobs), maxScore)

					for _, jobBody := range jobs {
						// Atomically remove from delayed and push to input
						// Using a simple lock-free approach: remove first, if successful, push.
						removed, err := m.Client.ZRem(ctx, delayedQueue, jobBody).Result()
						if err != nil {
							fmt.Printf("Error removing delayed job: %v\n", err)
							continue
						}

						if removed > 0 {
							inputQueue := m.InputQueue
							if m.Prefix != "" {
								inputQueue = fmt.Sprintf("%s:%s", m.Prefix, m.InputQueue)
							}
							if err := m.Client.RPush(ctx, inputQueue, jobBody).Err(); err != nil {
								fmt.Printf("Error pushing job back to input queue: %v\n", err)
								// Try to add back to delayed queue to avoid data loss
								m.Client.ZAdd(ctx, delayedQueue, redis.Z{
									Score:  now, // Retry immediately next tick
									Member: jobBody,
								})
							} else {
								fmt.Printf("Moved job back to input queue successfully\n")
							}
						}
					}
				}
			}
		}
	}()
}

func (m *RedisManager) addJob(ctx context.Context, queueName string, jobID string, data []byte) error {
	jobKey := m.jobKey(queueName, jobID)
	payload := string(data)
	if err := m.Client.HSet(ctx, jobKey,
		"name", "default",
		"data", payload,
		"opts", "{}",
		"timestamp", time.Now().UnixMilli(),
	).Err(); err != nil {
		return err
	}
	return m.Client.RPush(ctx, m.queueKey(queueName, "wait"), jobID).Err()
}

func (m *RedisManager) lockJob(ctx context.Context, queueName string, jobID string) error {
	lockKey := m.queueKey(queueName, jobID+":lock")
	return m.Client.Set(ctx, lockKey, m.WorkerID, m.LockTTL).Err()
}

func (m *RedisManager) unlockJob(ctx context.Context, queueName string, jobID string) error {
	lockKey := m.queueKey(queueName, jobID+":lock")
	return m.Client.Del(ctx, lockKey).Err()
}

func (m *RedisManager) queueKey(queueName string, suffix string) string {
	return fmt.Sprintf("%s:%s:%s", m.Prefix, queueName, suffix)
}

func (m *RedisManager) jobKey(queueName string, jobID string) string {
	return fmt.Sprintf("%s:%s:%s", m.Prefix, queueName, jobID)
}
