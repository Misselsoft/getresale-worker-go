package main

import (
	"bufio"
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"getresale-worker-go/internal/llm"
	"getresale-worker-go/internal/queue"
	"getresale-worker-go/internal/worker"

	"github.com/joho/godotenv"
	"github.com/redis/go-redis/v9"
)

// decidePendingJobsMode decide se os jobs já pendentes nas filas do Redis
// devem ser carregados (processados) ou ignorados (movidos para uma fila de
// backup) neste boot do worker. Se pendingFlag for "load" ou "skip", a
// decisão é imediata. Caso contrário, e se houver jobs pendentes, o usuário
// tem 10s para decidir interativamente; por padrão (timeout ou stdin sem
// resposta útil) os jobs pendentes são ignorados.
func decidePendingJobsMode(pendingFlag string, mainCount, oppCount int64) string {
	mode := strings.ToLower(strings.TrimSpace(pendingFlag))
	if mode == "load" || mode == "skip" {
		log.Printf("Pending jobs mode set via --pending=%s flag", mode)
		return mode
	}
	if mode != "" {
		log.Fatalf("Invalid --pending value %q: must be 'load' or 'skip'", pendingFlag)
	}

	if mainCount+oppCount == 0 {
		return "skip"
	}

	log.Printf("Found %d pending job(s) in input queue and %d in opportunity queue.", mainCount, oppCount)
	log.Println("Carregar jobs pendentes? Digite 's' e Enter para CARREGAR. Aguardando 10s (padrão: ignorar e mover para fila de backup)...")

	answer := make(chan string, 1)
	go func() {
		reader := bufio.NewReader(os.Stdin)
		line, _ := reader.ReadString('\n')
		answer <- strings.ToLower(strings.TrimSpace(line))
	}()

	select {
	case line := <-answer:
		if line == "s" || line == "sim" || line == "y" || line == "yes" || line == "load" {
			return "load"
		}
		return "skip"
	case <-time.After(10 * time.Second):
		log.Println("Tempo esgotado. Ignorando jobs pendentes por padrão.")
		return "skip"
	}
}

func main() {
	log.Println("Initializing GetResale Worker...")

	pendingFlag := flag.String("pending", "", "Como tratar jobs pendentes no Redis ao iniciar: 'load' (processar) ou 'skip' (mover para fila de backup e ignorar)")
	flag.Parse()

	// Load .env file
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, using system environment variables")
	} else {
		log.Println(".env file loaded successfully")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Env variables
	redisHost := os.Getenv("REDIS_HOST")
	if redisHost == "" {
		redisHost = "localhost"
	}
	redisPort := os.Getenv("REDIS_PORT")
	if redisPort == "" {
		redisPort = "6379"
	}
	redisPassword := os.Getenv("REDIS_PASSWORD")
	redisDBStr := os.Getenv("REDIS_DB")
	redisDB := 0
	if redisDBStr != "" {
		if value, err := strconv.Atoi(redisDBStr); err == nil && value >= 0 {
			redisDB = value
		}
	}

	inputQueue := os.Getenv("REDIS_INPUT_QUEUE")
	outputQueue := os.Getenv("REDIS_OUTPUT_QUEUE")
	geminiKey := os.Getenv("GEMINI_API_KEY")
	geminiKey = strings.TrimSpace(geminiKey)
	if geminiKey == "" {
		log.Println("WARNING: GEMINI_API_KEY is not set! LLM calls will fail.")
	} else {
		if len(geminiKey) > 4 {
			log.Printf("GEMINI_API_KEY loaded (len=%d, prefix=%s...)", len(geminiKey), geminiKey[:4])
		} else {
			log.Printf("GEMINI_API_KEY loaded (len=%d, too short to show prefix)", len(geminiKey))
		}
	}

	openAIKey := os.Getenv("OPENAI_API_KEY")
	openAIBaseURL := os.Getenv("OPENAI_BASE_URL")

	maxConcurrencyStr := os.Getenv("REDIS_MAX_CONCURRENCY")
	maxConcurrency, err := strconv.Atoi(maxConcurrencyStr)
	if err != nil || maxConcurrency <= 0 {
		maxConcurrency = 5 // Default
	}

	geminiModel := os.Getenv("GEMINI_MODEL")
	if geminiModel == "" {
		geminiModel = "gemini-3-flash-preview" // Default model
	}

	if inputQueue == "" || outputQueue == "" {
		log.Fatal("REDIS_INPUT_QUEUE and REDIS_OUTPUT_QUEUE must be set")
	}

	// Clients
	redisClient := redis.NewClient(&redis.Options{
		Addr:     redisHost + ":" + redisPort,
		Password: redisPassword,
		DB:       redisDB,
	})
	redisPrefix := os.Getenv("REDIS_PREFIX")
	workerID := os.Getenv("WORKER_ID")
	if workerID == "" {
		hostname, err := os.Hostname()
		if err != nil {
			workerID = "worker"
		} else {
			workerID = hostname
		}
		workerID = workerID + ":" + strconv.Itoa(os.Getpid())
	}
	redisManager := queue.NewRedisManager(redisClient, inputQueue, outputQueue, workerID, redisPrefix)

	// DEBUG: List all keys in Redis to verify queue names
	keys, err := redisClient.Keys(context.Background(), "*").Result()
	if err != nil {
		log.Printf("Error listing keys: %v\n", err)
	} else {
		log.Println("--- REDIS KEYS ---")
		for _, key := range keys {
			log.Println(key)
		}
		log.Println("------------------")
	}

	geminiClient := llm.NewGeminiClient(geminiKey, geminiModel)
	openAIClient := llm.NewOpenAIClient(openAIKey, openAIBaseURL)

	// Worker
	w := worker.NewWorker(redisManager, geminiClient, openAIClient, maxConcurrency)

	// Opportunity Worker
	oppQueueName := "opportunity_analysis_queue"
	// Output queue for LLM results (consumed by NestJS)
	oppOutputQueue := "LLM_OUTPUT"
	oppRedisManager := queue.NewRedisManager(redisClient, oppQueueName, oppOutputQueue, workerID+"-opp", redisPrefix)

	// Pending jobs control: by default, jobs already sitting in the queues are
	// NOT processed on startup (moved to a `:skipped` backup queue instead),
	// to avoid unexpected LLM token costs when restarting the worker locally.
	mainPending, err := redisManager.PendingCount(context.Background())
	if err != nil {
		log.Printf("Warning: could not check pending jobs in input queue: %v", err)
	}
	oppPending, err := oppRedisManager.PendingCount(context.Background())
	if err != nil {
		log.Printf("Warning: could not check pending jobs in opportunity queue: %v", err)
	}

	if mode := decidePendingJobsMode(*pendingFlag, mainPending, oppPending); mode == "skip" {
		if moved, err := redisManager.SkipPendingToBackup(context.Background()); err != nil {
			log.Printf("Warning: failed to move pending input-queue jobs to backup: %v", err)
		} else if moved > 0 {
			log.Printf("Moved %d pending job(s) from input queue to backup (will not be processed this run).", moved)
		}
		if moved, err := oppRedisManager.SkipPendingToBackup(context.Background()); err != nil {
			log.Printf("Warning: failed to move pending opportunity-queue jobs to backup: %v", err)
		} else if moved > 0 {
			log.Printf("Moved %d pending job(s) from opportunity queue to backup (will not be processed this run).", moved)
		}
	} else {
		log.Println("Loading pending jobs as requested.")
	}

	// Opportunity Worker no longer needs DB, just Gemini
	oppWorker := worker.NewOpportunityWorker(oppRedisManager, geminiClient, maxConcurrency, geminiModel)

	// Start Opportunity Worker
	go oppWorker.Start(ctx)

	// Start
	w.Start(ctx)

	log.Println("Worker shut down gracefully.")
}
