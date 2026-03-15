import { useEffect, useRef } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChatMessage, type Message } from "./chat-message"
import { Bot, Sparkles } from "lucide-react"

interface ChatViewportProps {
  messages: Message[]
}

export function ChatViewport({ messages }: ChatViewportProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  if (messages.length === 0) {
    return <EmptyState />
  }

  return (
    <ScrollArea className="flex-1 px-4">
      <div className="max-w-4xl mx-auto py-8 space-y-6">
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  )
}

function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4">
      <div className="max-w-2xl w-full text-center space-y-8">
        {/* Logo/Icon */}
        <div className="relative mx-auto w-20 h-20">
          <div className="absolute inset-0 bg-primary/20 rounded-full blur-2xl" />
          <div className="relative flex items-center justify-center w-full h-full rounded-full bg-secondary border border-border">
            <Bot className="h-10 w-10 text-primary" />
          </div>
        </div>

        {/* Welcome Text */}
        <div className="space-y-3">
          <h1 className="text-3xl font-semibold text-foreground tracking-tight text-balance">
            Welcome to Autara AI
          </h1>
          <p className="text-muted-foreground text-lg leading-relaxed max-w-md mx-auto text-pretty">
            Your intelligent assistant for conversations, document Q&A, and research.
          </p>
        </div>

        {/* Capabilities */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4">
          {capabilities.map((cap, i) => (
            <div
              key={i}
              className="p-4 rounded-xl bg-secondary/50 border border-border/50 text-left"
            >
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium text-foreground">{cap.title}</span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {cap.description}
              </p>
            </div>
          ))}
        </div>

        {/* Suggestions */}
        <div className="pt-4">
          <p className="text-xs text-muted-foreground mb-3">Try asking:</p>
          <div className="flex flex-wrap justify-center gap-2">
            {suggestions.map((suggestion, i) => (
              <button
                key={i}
                className="px-4 py-2 text-sm text-foreground/80 bg-secondary/50 hover:bg-secondary border border-border/50 rounded-full transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

const capabilities = [
  {
    title: "Document Q&A",
    description: "Upload Markdown files for Q&A.",
  },
  {
    title: "(WIP) Deep Reasoning",
    description: "(WIP) Enable step-by-step reasoning for complex problem solving.",
  },
  {
    title: "(WIP) Web Research",
    description: "(WIP) Search the web in real-time to find current information.",
  },
]

const suggestions = [
  "Explain quantum computing",
  "Review my code",
  "Summarize a document",
  "Help me brainstorm",
]
