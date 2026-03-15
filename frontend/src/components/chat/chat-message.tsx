import { cn } from "@/lib/utils"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Bot, User } from "lucide-react"
import ReactMarkdown from "react-markdown"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  isStreaming?: boolean
}

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"

  return (
    <div
      className={cn(
        "flex gap-4 animate-fade-in-up",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <Avatar className={cn(
        "h-10 w-10 shrink-0 border-2",
        isUser
          ? "border-primary/30 bg-primary/10"
          : "border-accent/30 bg-accent/10"
      )}>
        <AvatarFallback className={cn(
          "text-foreground",
          isUser ? "bg-message-user" : "bg-message-ai"
        )}>
          {isUser ? (
            <User className="h-5 w-5" />
          ) : (
            <Bot className="h-5 w-5 text-primary" />
          )}
        </AvatarFallback>
      </Avatar>

      <div
        className={cn(
          "max-w-[75%] rounded-2xl px-5 py-4",
          isUser
            ? "bg-message-user border border-border/50"
            : "bg-message-ai border border-border/30"
        )}
      >
        {message.isStreaming && !message.content ? (
          <TypingIndicator />
        ) : (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown
              components={{
                p: ({ children }) => (
                  <p className="text-foreground/90 leading-relaxed mb-2 last:mb-0">
                    {children}
                  </p>
                ),
                code: ({ children, className }) => {
                  const isInline = !className
                  return isInline ? (
                    <code className="bg-secondary px-1.5 py-0.5 rounded text-primary font-mono text-sm">
                      {children}
                    </code>
                  ) : (
                    <code className="block bg-secondary p-4 rounded-lg overflow-x-auto font-mono text-sm text-foreground/90">
                      {children}
                    </code>
                  )
                },
                pre: ({ children }) => (
                  <pre className="bg-secondary rounded-lg overflow-hidden my-3">
                    {children}
                  </pre>
                ),
                ul: ({ children }) => (
                  <ul className="list-disc list-inside space-y-1 text-foreground/90 my-2">
                    {children}
                  </ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-inside space-y-1 text-foreground/90 my-2">
                    {children}
                  </ol>
                ),
                h1: ({ children }) => (
                  <h1 className="text-xl font-semibold text-foreground mb-3">{children}</h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-lg font-semibold text-foreground mb-2">{children}</h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-base font-semibold text-foreground mb-2">{children}</h3>
                ),
                a: ({ children, href }) => (
                  <a
                    href={href}
                    className="text-primary hover:underline"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {children}
                  </a>
                ),
                blockquote: ({ children }) => (
                  <blockquote className="border-l-2 border-primary/50 pl-4 italic text-muted-foreground my-3">
                    {children}
                  </blockquote>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
            {message.isStreaming && message.content && (
              <span className="inline-block w-2 h-5 bg-primary ml-1 animate-pulse" />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 bg-primary/60 rounded-full animate-typing-dot"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  )
}
