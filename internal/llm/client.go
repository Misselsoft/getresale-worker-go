package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

type RetryableError struct {
	RetryDelay time.Duration
	Err        error
}

func (e *RetryableError) Error() string {
	return e.Err.Error()
}

type TokenUsage struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CandidateTokens  int     `json:"candidate_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	CachedTokens     int     `json:"cached_tokens"`
	EstimatedCostUSD float64 `json:"estimated_cost_usd"`
}

type Client interface {
	Generate(ctx context.Context, model, prompt, format string) (string, TokenUsage, error)
}

// OllamaClient
type OllamaClient struct {
	BaseURL string
	HTTP    *http.Client
}

func NewOllamaClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		BaseURL: baseURL,
		HTTP:    &http.Client{Timeout: 5 * time.Minute},
	}
}

type ollamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
	Format string `json:"format,omitempty"`
}

type ollamaResponse struct {
	Response string `json:"response"`
}

func (c *OllamaClient) Generate(ctx context.Context, model, prompt, format string) (string, TokenUsage, error) {
	reqBody := ollamaRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
		Format: format,
	}
	data, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/api/generate", c.BaseURL), bytes.NewBuffer(data))
	if err != nil {
		return "", TokenUsage{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	var resp *http.Response
	maxRetries := 3
	baseDelay := 1 * time.Second

	for i := 0; i <= maxRetries; i++ {
		if i > 0 {
			time.Sleep(baseDelay * time.Duration(1<<uint(i-1))) // Exponential backoff: 1s, 2s, 4s
			log.Printf("Retrying Gemini request (attempt %d/%d) after error...", i+1, maxRetries+1)
		}

		resp, err = c.HTTP.Do(req)
		if err != nil {
			// Network error, retry
			log.Printf("Gemini network error: %v", err)
			continue
		}

		if resp.StatusCode >= 500 {
			// Server error, retry
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			log.Printf("Gemini server error (status %d): %s", resp.StatusCode, string(body))
			req.Body = io.NopCloser(bytes.NewBuffer(data))
			continue
		}

		// If success or non-retriable error (e.g. 400, 401, 429 handled later), break
		break
	}

	if err != nil {
		return "", TokenUsage{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", TokenUsage{}, fmt.Errorf("ollama error (status %d): %s", resp.StatusCode, string(body))
	}

	var res ollamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", TokenUsage{}, err
	}

	return res.Response, TokenUsage{}, nil
}

// GeminiClient
type GeminiClient struct {
	APIKey       string
	BaseURL      string
	DefaultModel string
	HTTP         *http.Client
}

func NewGeminiClient(apiKey, defaultModel string) *GeminiClient {
	if defaultModel == "" {
		defaultModel = "gemini-3-flash-preview"
	}
	return &GeminiClient{
		APIKey:       apiKey,
		BaseURL:      "https://generativelanguage.googleapis.com/v1beta",
		DefaultModel: defaultModel,
		HTTP:         &http.Client{Timeout: 5 * time.Minute},
	}
}

type GeminiPart struct {
	Text string `json:"text"`
}

type GeminiContent struct {
	Parts []GeminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type geminiGenerationConfig struct {
	ResponseMimeType string `json:"responseMimeType,omitempty"`
}

type geminiRequest struct {
	Contents         []GeminiContent         `json:"contents,omitempty"`
	GenerationConfig *geminiGenerationConfig `json:"generationConfig,omitempty"`
	CachedContent    string                  `json:"cachedContent,omitempty"`
}

type geminiCandidate struct {
	Content GeminiContent `json:"content"`
}

type geminiUsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
}

type geminiResponse struct {
	Candidates    []geminiCandidate   `json:"candidates"`
	UsageMetadata geminiUsageMetadata `json:"usageMetadata"`
}

type geminiErrorResponse struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
		Details []struct {
			Type       string `json:"@type"`
			RetryDelay string `json:"retryDelay,omitempty"`
		} `json:"details"`
	} `json:"error"`
}

type GeminiCachedContent struct {
	Name        string          `json:"name,omitempty"`
	DisplayName string          `json:"displayName"`
	Model       string          `json:"model"`
	Contents    []GeminiContent `json:"contents"`
	TTL         string          `json:"ttl"`
	ExpireTime  string          `json:"expireTime,omitempty"`
}

func (c *GeminiClient) CreateCache(ctx context.Context, model, displayName string, contents []GeminiContent, ttlSeconds int) (*GeminiCachedContent, error) {
	reqBody := GeminiCachedContent{
		Model:       fmt.Sprintf("models/%s", model),
		DisplayName: displayName,
		Contents:    contents,
		TTL:         fmt.Sprintf("%ds", ttlSeconds),
	}
	data, _ := json.Marshal(reqBody)

	url := fmt.Sprintf("%s/cachedContents?key=%s", c.BaseURL, c.APIKey)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini create cache error (status %d): %s", resp.StatusCode, string(body))
	}

	var res GeminiCachedContent
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}

	return &res, nil
}

func (c *GeminiClient) Generate(ctx context.Context, model, prompt, format string) (string, TokenUsage, error) {
	return c.GenerateWithCache(ctx, model, prompt, format, "")
}

func (c *GeminiClient) GenerateWithCache(ctx context.Context, model, prompt, format, cachedContentName string) (string, TokenUsage, error) {
	// Use default model if not provided or if it's a legacy Llama model name
	if model == "" || model == "llama3.2" || model == "llama2" {
		model = c.DefaultModel
	}

	reqBody := geminiRequest{
		Contents: []GeminiContent{
			{
				Parts: []GeminiPart{
					{Text: prompt},
				},
				Role: "user",
			},
		},
	}

	if cachedContentName != "" {
		reqBody.CachedContent = cachedContentName
	}

	if format == "json" {
		reqBody.GenerationConfig = &geminiGenerationConfig{
			ResponseMimeType: "application/json",
		}
	}

	data, _ := json.Marshal(reqBody)

	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", c.BaseURL, model, c.APIKey)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(data))
	if err != nil {
		return "", TokenUsage{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	var resp *http.Response
	maxRetries := 3
	baseDelay := 1 * time.Second

	for i := 0; i <= maxRetries; i++ {
		if i > 0 {
			time.Sleep(baseDelay * time.Duration(1<<uint(i-1))) // Exponential backoff: 1s, 2s, 4s
			// Reset request body for retry
			req.Body = io.NopCloser(bytes.NewBuffer(data))
		}

		resp, err = c.HTTP.Do(req)
		if err != nil {
			// Network error, retry
			if i < maxRetries {
				continue
			}
			return "", TokenUsage{}, err
		}

		if resp.StatusCode >= 500 {
			// Server error, retry
			if i < maxRetries {
				resp.Body.Close()
				continue
			}
			// If max retries reached, let it fall through to error handling below
		}

		// If success or non-retriable error (e.g. 400, 401, 429 handled later), break
		break
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		// Redact key from URL for logging
		redactedURL := strings.ReplaceAll(url, c.APIKey, "REDACTED")

		baseErr := fmt.Errorf("gemini error (status %d) at %s: %s", resp.StatusCode, redactedURL, string(body))

		if resp.StatusCode == http.StatusTooManyRequests {
			var errResp geminiErrorResponse
			if err := json.Unmarshal(body, &errResp); err == nil {
				for _, detail := range errResp.Error.Details {
					if detail.RetryDelay != "" {
						if d, err := time.ParseDuration(detail.RetryDelay); err == nil {
							return "", TokenUsage{}, &RetryableError{
								RetryDelay: d,
								Err:        baseErr,
							}
						}
					}
				}
			}
			// If we couldn't parse delay but it's 429, default to a reasonable backoff
			return "", TokenUsage{}, &RetryableError{
				RetryDelay: 30 * time.Second,
				Err:        baseErr,
			}
		}

		return "", TokenUsage{}, baseErr
	}

	var res geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", TokenUsage{}, err
	}

	if len(res.Candidates) == 0 || len(res.Candidates[0].Content.Parts) == 0 {
		return "", TokenUsage{}, fmt.Errorf("gemini returned no content")
	}

	usage := TokenUsage{
		PromptTokens:    res.UsageMetadata.PromptTokenCount,
		CandidateTokens: res.UsageMetadata.CandidatesTokenCount,
		TotalTokens:     res.UsageMetadata.TotalTokenCount,
		CachedTokens:    res.UsageMetadata.CachedContentTokenCount,
	}

	// Calculate cost (Gemini 1.5 Flash Pricing as proxy)
	// Input: $0.075 / 1M
	// Output: $0.30 / 1M
	// Cached Input: $0.01875 / 1M
	inputCostPerMillion := 0.075
	outputCostPerMillion := 0.30
	cachedInputCostPerMillion := 0.01875

	// Adjust for other models if needed
	if strings.Contains(model, "pro") {
		// Gemini 1.5 Pro
		// Input: $3.50 / 1M
		// Output: $10.50 / 1M
		// Cached Input: $0.875 / 1M
		inputCostPerMillion = 3.50
		outputCostPerMillion = 10.50
		cachedInputCostPerMillion = 0.875
	}

	usage.EstimatedCostUSD = (float64(usage.PromptTokens)/1_000_000)*inputCostPerMillion +
		(float64(usage.CandidateTokens)/1_000_000)*outputCostPerMillion +
		(float64(usage.CachedTokens)/1_000_000)*cachedInputCostPerMillion

	return res.Candidates[0].Content.Parts[0].Text, usage, nil
}

// OpenAIClient (Simple HTTP implementation to avoid extra dependencies for now)
type OpenAIClient struct {
	APIKey  string
	BaseURL string
	HTTP    *http.Client
}

func NewOpenAIClient(apiKey, baseURL string) *OpenAIClient {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	return &OpenAIClient{
		APIKey:  apiKey,
		BaseURL: baseURL,
		HTTP:    &http.Client{Timeout: 5 * time.Minute},
	}
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIRequest struct {
	Model    string          `json:"model"`
	Messages []openAIMessage `json:"messages"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAIChoice struct {
	Message openAIMessage `json:"message"`
}

type openAIResponse struct {
	Choices []openAIChoice `json:"choices"`
	Usage   openAIUsage    `json:"usage"`
}

func (c *OpenAIClient) Generate(ctx context.Context, model, prompt, format string) (string, TokenUsage, error) {
	reqBody := openAIRequest{
		Model: model,
		Messages: []openAIMessage{
			{Role: "user", Content: prompt},
		},
	}
	data, _ := json.Marshal(reqBody)

	req, _ := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/chat/completions", c.BaseURL), bytes.NewBuffer(data))
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.APIKey))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return "", TokenUsage{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", TokenUsage{}, fmt.Errorf("openai error (status %d): %s", resp.StatusCode, string(body))
	}

	var res openAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", TokenUsage{}, err
	}

	if len(res.Choices) == 0 {
		return "", TokenUsage{}, fmt.Errorf("openai returned no choices")
	}

	usage := TokenUsage{
		PromptTokens:    res.Usage.PromptTokens,
		CandidateTokens: res.Usage.CompletionTokens,
		TotalTokens:     res.Usage.TotalTokens,
	}

	// Cost calculation for OpenAI
	var inputCost, outputCost float64
	if model == "gpt-4o" {
		inputCost = 5.00
		outputCost = 15.00
	} else if model == "gpt-4" {
		inputCost = 30.00
		outputCost = 60.00
	} else {
		// GPT-3.5 Turbo or others
		inputCost = 0.50
		outputCost = 1.50
	}

	usage.EstimatedCostUSD = (float64(usage.PromptTokens)/1_000_000)*inputCost +
		(float64(usage.CandidateTokens)/1_000_000)*outputCost

	return res.Choices[0].Message.Content, usage, nil
}
