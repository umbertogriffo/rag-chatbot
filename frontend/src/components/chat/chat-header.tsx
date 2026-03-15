import { Button } from "@/components/ui/button"
import { Bot, Plus, History, Settings } from "lucide-react"

interface ChatHeaderProps {
  onNewChat: () => void
  disabled?: boolean
}

export function ChatHeader({ onNewChat, disabled }: ChatHeaderProps) {
  return (
    <header className="shrink-0 h-16 border-b border-border/50 bg-background/80 backdrop-blur-sm">
      <div className="h-full max-w-7xl mx-auto px-4 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-primary/30 rounded-lg blur-md" />
            <div className="relative flex items-center justify-center w-9 h-9 rounded-lg bg-secondary border border-border">
              <Bot className="h-5 w-5 text-primary" />
            </div>
          </div>
          <div>
            <h1 className="font-semibold text-foreground tracking-tight">Autara AI</h1>
            <p className="text-xs text-muted-foreground">Powered by llamacpp and Chroma</p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
          >
            <History className="h-4 w-4 mr-2" />
            <span className="hidden sm:inline">History</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onNewChat}
            disabled={disabled}
            className="text-muted-foreground hover:text-foreground"
          >
            <Plus className="h-4 w-4 mr-2" />
            <span className="hidden sm:inline">New Chat</span>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="text-muted-foreground hover:text-foreground"
          >
            <Settings className="h-4 w-4" />
            <span className="sr-only">Settings</span>
          </Button>
        </div>
      </div>
    </header>
  )
}
