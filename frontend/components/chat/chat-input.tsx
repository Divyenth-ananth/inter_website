'use client'

import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Attachment } from '@/hooks/use-chat'

interface ChatInputProps {
  onSendMessage: (message: string, attachments?: Attachment[]) => void
  isLoading: boolean
}

export default function ChatInput({
  onSendMessage,
  isLoading,
}: ChatInputProps) {
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  /* ------------------------------------------------
      FILE UPLOAD HANDLER
  -------------------------------------------------- */
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (!files) return

    Array.from(files).forEach((file) => {
      const reader = new FileReader()
      reader.onload = (event) => {
        const attachment: Attachment = {
          id: `file-${Date.now()}-${Math.random()}`,
          name: file.name,
          type: file.type,
          size: file.size,
          url: event.target?.result as string,
          previewUrl: file.type.startsWith('image/') ? (event.target?.result as string) : undefined,
          file: file,      // ✔ IMPORTANT — real file object
        }

        setAttachments((prev) => [...prev, attachment])
      }
      reader.readAsDataURL(file)
    })

    // allow re-upload of same file
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const removeAttachment = (id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id))
  }

  /* ------------------------------------------------
      FORM SUBMISSION
  -------------------------------------------------- */
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (isLoading) return
    if (!input.trim() && attachments.length === 0) return

    onSendMessage(
      input || (attachments.length > 0 ? `Shared ${attachments.length} file(s)` : ''),
      attachments
    )

    // reset
    setInput('')
    setAttachments([])
    inputRef.current?.focus()
  }

  /* ------------------------------------------------
      UI COMPONENT
  -------------------------------------------------- */
  return (
    <div className="border-t border-border p-4 lg:p-6 bg-background">

      {/* ----------- PREVIEW BAR ------------- */}
      {attachments.length > 0 && (
        <div className="mb-4 max-w-4xl mx-auto">
          <div className="flex flex-wrap gap-2">
            {attachments.map((attachment) => (
              <div
                key={attachment.id}
                className="relative group bg-card border border-border rounded-lg p-2 hover:border-primary transition-colors"
              >
                {attachment.previewUrl ? (
                  <img
                    src={attachment.previewUrl}
                    alt={attachment.name}
                    className="h-16 w-16 object-cover rounded"
                  />
                ) : (
                  <div className="h-16 w-16 flex items-center justify-center bg-muted rounded">
                    <span className="text-xs text-muted-foreground truncate">
                      {attachment.name.split('.').pop()?.toUpperCase()}
                    </span>
                  </div>
                )}

                {/* remove button */}
                <button
                  onClick={() => removeAttachment(attachment.id)}
                  className="absolute -top-2 -right-2 bg-destructive text-destructive-foreground 
                             rounded-full w-5 h-5 flex items-center justify-center text-xs 
                             opacity-0 group-hover:opacity-100 transition-opacity"
                  type="button"
                >
                  ×
                </button>

                <p className="text-xs text-muted-foreground mt-1 max-w-16 truncate">
                  {attachment.name}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ----------- INPUT ROW ------------- */}
      <form onSubmit={handleSubmit} className="flex gap-3 max-w-4xl mx-auto">

        {/* Hidden input */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*,.pdf,.doc,.docx,.txt,.xls,.xlsx"
          onChange={handleFileSelect}
          className="hidden"
        />

        <Button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          variant="outline"
          className="bg-card border-border hover:bg-muted text-foreground"
        >
          📎
        </Button>

        <Input
          ref={inputRef}
          type="text"
          placeholder="Ask about what you see..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
          className="bg-card border-border text-foreground placeholder:text-muted-foreground flex-1"
        />

        <Button
          type="submit"
          disabled={isLoading || (!input.trim() && attachments.length === 0)}
          className="bg-primary hover:bg-primary/90 text-primary-foreground disabled:opacity-50"
        >
          {isLoading ? '⏳' : '→'}
        </Button>
      </form>
    </div>
  )
}

