'use client'

import { useState, useEffect, useRef } from 'react'
import ChatSidebar from '@/components/chat/chat-sidebar'
import ChatWindow from '@/components/chat/chat-window'
import ChatInput from '@/components/chat/chat-input'
import ChatHeader from '@/components/chat/chat-header'
import { useChat } from '@/hooks/use-chat'

export default function ChatPage() {
  const [user, setUser] = useState<{ fullName?: string; email: string } | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [showSettings, setShowSettings] = useState(false)

  useEffect(() => {
    const userData = localStorage.getItem('user')
    if (!userData) {
      window.location.href = '/'
      return
    }
    setUser(JSON.parse(userData))
  }, [])

  const {
    sessions,
    currentSession,
    messages,
    isLoading,
    createSession,
    selectSession,
    deleteSession,
    sendMessage,
    clearConversations,
  } = useChat()



  const handleSendMessage = (message: string, attachments?: any[]) => {
    sendMessage(message, attachments)
  }

  if (!user) return null

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <ChatSidebar
        sessions={sessions}
        currentSession={currentSession}
        onNewSession={createSession}
        onSelectSession={selectSession}
        onDeleteSession={deleteSession}
        onClearAll={clearConversations}
        user={user}
        open={sidebarOpen}
        onOpenChange={setSidebarOpen}
      />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header with theme toggle and settings */}
        <ChatHeader 
          showSettings={showSettings}
          onSettingsChange={setShowSettings}
        />

        {/* Chat window */}
        <ChatWindow messages={messages} isLoading={isLoading} />

        {/* Input area */}
        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  )
}
