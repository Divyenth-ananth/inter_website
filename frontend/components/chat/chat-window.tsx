'use client'

import { useEffect, useRef , useState } from 'react'
import { Card } from '@/components/ui/card'
import { Attachment } from '@/hooks/use-chat'
import ImageModal from "@/components/ImageModal";


interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  attachments?: Attachment[]
}

interface ChatWindowProps {
  messages: Message[]
  isLoading: boolean
}

export default function ChatWindow({ messages, isLoading }: ChatWindowProps) {
  
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  return (
    <div className="flex-1 overflow-y-auto p-4 lg:p-6 space-y-4">
      {messages.length === 0 ? (
        <div className="h-full flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <div className="text-5xl mb-4">🛰️</div>
            <p className="text-lg font-medium text-foreground mb-2">
              Chat about this image
            </p>
            <p className="text-sm">
              Upload an image or ask questions about satellite imagery
            </p>
          </div>
        </div>
      ) : (
        <>
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className="max-w-xs lg:max-w-md xl:max-w-lg">

                {/* ⭐ CHANGED: Moved attachments HERE (above bubble) */}
                {message.attachments && message.attachments.length > 0 && (
                  <div className={`flex flex-wrap gap-2 mb-2 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {message.attachments.map((attachment) => (
                      <div
                        key={attachment.id}
                        className="rounded-lg overflow-hidden border border-border bg-card"
                      >
                        {attachment.previewUrl ? (
                          <img
                            src={attachment.previewUrl || attachment.url}
                            alt={attachment.name}
                            className="h-80 w-80 object-cover cursor-pointer hover:opacity-80"
                           onClick={() => {
                            setModalImage(attachment.previewUrl || attachment.url);
                            setModalOpen(true);
                          }}
                          />

                        ) : (
                          <div className="h-32 w-32 flex items-center justify-center bg-muted">
                            <div className="text-center">
                              <div className="text-2xl mb-1">📄</div>
                              <p className="text-xs text-muted-foreground max-w-20 truncate">
                                {attachment.name}
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {/* ⭐ END OF CHANGE */}

                {/* --- ORIGINAL MESSAGE BUBBLE (unchanged) --- */}
                <Card
                  className={`p-4 ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground rounded-3xl rounded-tr-sm'
                      : 'bg-card border-border rounded-3xl rounded-tl-sm'
                  }`}
                >
                  <p className="text-sm break-words">{message.content}</p>
                  <p
                    className={`text-xs mt-2 opacity-70 ${
                      message.role === 'user'
                        ? 'text-primary-foreground'
                        : 'text-muted-foreground'
                    }`}
                  >
                    {new Date(message.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </p>
                </Card>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <Card className="bg-card border-border p-4 rounded-3xl rounded-tl-sm">
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                </div>
              </Card>
            </div>
          )}

          <div ref={scrollRef} />
        </>
      )}
      <ImageModal
        isOpen={modalOpen}
        imageUrl={modalImage}
        onClose={() => setModalOpen(false)}
      />

    </div>
  )
}
