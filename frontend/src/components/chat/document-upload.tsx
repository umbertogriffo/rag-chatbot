import { useState, useCallback } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Upload, X, FileText, File, ImageIcon, FileCode } from "lucide-react"
import { uploadDocument, deleteDocument, type DocumentInfo } from "@/services/api"
import type { UploadProgress } from "@/hooks/useDocuments"

interface DocumentUploadProps {
  documents: DocumentInfo[]
  uploading: UploadProgress | null
  onDocumentsChange: (updater: (prev: DocumentInfo[]) => DocumentInfo[]) => void
  onUploadStart: (filename: string) => void
  onUploadProgress: (filename: string, progress: number) => void
  onUploadEnd: () => void
  onError: (message: string) => void
  isExpanded: boolean
  onToggleExpand: () => void
}

export function DocumentUpload({
  documents,
  uploading,
  onDocumentsChange,
  onUploadStart,
  onUploadProgress,
  onUploadEnd,
  onError,
  isExpanded,
  onToggleExpand,
}: DocumentUploadProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    files.forEach(f => handleUploadFile(f))
  }, [])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    files.forEach(f => handleUploadFile(f))
    // Reset the input so re-selecting the same file triggers onChange
    e.target.value = ""
  }, [])

  const handleUploadFile = async (file: File) => {
    onUploadStart(file.name)
    try {
      const result = await uploadDocument(file, (pct) => {
        onUploadProgress(file.name, pct)
      })
      // Add the new document to the list
      onDocumentsChange(prev => [
        ...prev,
        {
          document_id: result.document_id,
          filename: result.filename,
          size: file.size,
          content_type: file.type,
        },
      ])
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Upload failed"
      onError(msg)
    } finally {
      onUploadEnd()
    }
  }

  const removeDocument = async (id: string) => {
    try {
      await deleteDocument(id)
      onDocumentsChange(prev => prev.filter(d => d.document_id !== id))
    } catch {
      onError("Failed to delete document")
    }
  }

  const getFileIcon = (type: string) => {
    if (type.startsWith("image/")) return <ImageIcon className="h-4 w-4" />
    if (type.includes("pdf")) return <FileText className="h-4 w-4" />
    if (type.includes("code") || type.includes("javascript") || type.includes("typescript"))
      return <FileCode className="h-4 w-4" />
    return <File className="h-4 w-4" />
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  if (!isExpanded && documents.length === 0) {
    return (
      <Button
        variant="ghost"
        size="sm"
        onClick={onToggleExpand}
        className="text-muted-foreground hover:text-foreground hover:bg-secondary/50"
      >
        <Upload className="h-4 w-4 mr-2" />
        Upload documents
      </Button>
    )
  }

  return (
    <div className="animate-fade-in-up">
      {isExpanded && (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            "relative border-2 border-dashed rounded-xl p-6 transition-all duration-200",
            isDragging
              ? "border-primary bg-primary/5"
              : "border-border/50 hover:border-border"
          )}
        >
          <input
            type="file"
            multiple
            accept=".md,.txt,.pdf,.html"
            onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <div className="flex flex-col items-center gap-2 text-center pointer-events-none">
            <div className={cn(
              "p-3 rounded-full transition-colors",
              isDragging ? "bg-primary/20" : "bg-secondary"
            )}>
              <Upload className={cn(
                "h-6 w-6 transition-colors",
                isDragging ? "text-primary" : "text-muted-foreground"
              )} />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">
                Drag & drop files here
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                or click to browse &middot; .md .txt .pdf .html
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Upload progress */}
      {uploading && (
        <div className="mt-3 flex items-center gap-3 p-3 rounded-lg bg-secondary/50 border border-border/30">
          <div className="p-2 rounded-md bg-secondary text-muted-foreground">
            <File className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-foreground truncate">
              {uploading.filename}
            </p>
            <Progress value={uploading.progress} className="h-1 mt-1.5" />
          </div>
        </div>
      )}

      {documents.length > 0 && (
        <div className="mt-3 space-y-2">
          {documents.map(doc => (
            <div
              key={doc.document_id}
              className="flex items-center gap-3 p-3 rounded-lg bg-secondary/50 border border-border/30"
            >
              <div className="p-2 rounded-md bg-secondary text-muted-foreground">
                {getFileIcon(doc.content_type)}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">
                  {doc.filename}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs text-muted-foreground">
                    {formatFileSize(doc.size)}
                  </span>
                  <span className="text-xs text-primary">Ready</span>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => removeDocument(doc.document_id)}
                className="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-destructive/10"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
      )}

      {(isExpanded || documents.length > 0) && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggleExpand}
          className="mt-2 text-xs text-muted-foreground hover:text-foreground"
        >
          {isExpanded ? "Collapse" : "Add more"}
        </Button>
      )}
    </div>
  )
}
