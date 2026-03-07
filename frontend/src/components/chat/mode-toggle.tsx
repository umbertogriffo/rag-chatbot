import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Database, Brain, Globe } from "lucide-react"

export interface ChatModes {
  rag: boolean
  reasoning: boolean
  webSearch: boolean
}

interface ModeToggleProps {
  modes: ChatModes
  onModesChange: (modes: ChatModes) => void
}

const modeConfig = [
  {
    key: "rag" as const,
    icon: Database,
    label: "RAG Mode",
    description: "Use uploaded documents for context",
  },
  {
    key: "reasoning" as const,
    icon: Brain,
    label: "Reasoning",
    description: "Enable step-by-step reasoning",
  },
  {
    key: "webSearch" as const,
    icon: Globe,
    label: "Web Search",
    description: "Search the web for answers",
  },
]

export function ModeToggle({ modes, onModesChange }: ModeToggleProps) {
  const toggleMode = (key: keyof ChatModes) => {
    onModesChange({ ...modes, [key]: !modes[key] })
  }

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex items-center gap-1">
        {modeConfig.map(({ key, icon: Icon, label, description }) => (
          <Tooltip key={key}>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => toggleMode(key)}
                className={cn(
                  "h-8 px-2.5 gap-1.5 text-xs font-medium transition-all duration-200",
                  modes[key]
                    ? "bg-primary/15 text-primary border border-primary/30 hover:bg-primary/20"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">{label}</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top" className="bg-popover border-border">
              <p className="font-medium">{label}</p>
              <p className="text-xs text-muted-foreground">{description}</p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </TooltipProvider>
  )
}
