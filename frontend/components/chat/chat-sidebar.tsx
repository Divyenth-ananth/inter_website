'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import Link from 'next/link'
import { useEffect } from "react";


interface ChatSession {
  id: string
  title: string
  createdAt: Date
}

interface ChatSidebarProps {
  sessions: ChatSession[]
  currentSession: ChatSession | null
  onNewSession: () => void
  onSelectSession: (id: string) => void
  onDeleteSession: (id: string) => void
  onClearAll: () => void
  user: { fullName?: string; email: string }
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function ChatSidebar({
  sessions,
  currentSession,
  onNewSession,
  onSelectSession,
  onDeleteSession,
  onClearAll,
  user,
  open,
  onOpenChange,
}: ChatSidebarProps) {
  const [showConfirm, setShowConfirm] = useState(false)

  // useEffect(() => {
  // if (!currentSession && sessions.length === 0) {
  //   onNewSession();   // auto create exactly ONE session
  // }
  // }, [currentSession, sessions]);


  return (
    <>
      {/* Mobile menu button */}
      <button
        onClick={() => onOpenChange(!open)}
        className="fixed top-4 left-4 z-40 lg:hidden p-2 hover:bg-secondary rounded-lg text-foreground"
      >
        ☰
      </button>

      {/* Sidebar */}
      <div
        className={`fixed lg:static inset-y-0 left-0 w-64 bg-sidebar border-r border-sidebar-border flex flex-col transform transition-transform duration-200 z-30 ${
          open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Header with logo */}
        <div className="p-4 border-b border-sidebar-border">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-primary-foreground text-lg">🛰</span>
            </div>
            <h1 className="font-bold text-foreground">GeoNLI</h1>
          </div>
        </div>

        {/* New session button */}
        <div className="p-4 border-b border-sidebar-border">
          <Button
            onClick={onNewSession}
            className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
          >
            + New Session
          </Button>
        </div>

        {/* Sessions list */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Sessions
          </p>
          {sessions.length === 0 ? (
            <p className="text-xs text-muted-foreground">No sessions yet</p>
          ) : (
            sessions.map((session) => (
              <div
                key={session.id}
                className={`p-3 rounded-lg cursor-pointer transition-colors ${
                  currentSession?.id === session.id
                    ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                    : 'text-sidebar-foreground hover:bg-sidebar-accent'
                }`}
                onClick={() => onSelectSession(session.id)}
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm truncate flex-1">{session.title}</p>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteSession(session.id)
                    }}
                    className="text-xs opacity-0 group-hover:opacity-100 hover:text-destructive"
                  >
                    ✕
                  </button>
                </div>
                <p className="text-xs opacity-60 mt-1">
                  {new Date(session.createdAt).toLocaleDateString()}
                </p>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-sidebar-border p-4 space-y-2">
          <Button
            variant="outline"
            onClick={() => setShowConfirm(true)}
            className="w-full text-xs"
          >
            🗑️ Clear All
          </Button>
          <Link href="/" className="block">
            <Button
              variant="outline"
              className="w-full text-xs"
              onClick={() => localStorage.removeItem('user')}
            >
              Logout
            </Button>
          </Link>

          {/* User info */}
          <div className="pt-4 border-t border-sidebar-border">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center text-xs font-bold text-primary-foreground">
                {user.fullName?.charAt(0) || user.email.charAt(0)}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-sidebar-foreground truncate">
                  {user.fullName || 'User'}
                </p>
                <p className="text-xs text-muted-foreground truncate">{user.email}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Clear confirmation modal */}
        {showConfirm && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <Card className="bg-card border-border p-6 max-w-xs">
              <p className="text-foreground font-medium mb-4">
                Clear all conversations?
              </p>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => setShowConfirm(false)}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => {
                    onClearAll()
                    setShowConfirm(false)
                  }}
                  className="flex-1 bg-destructive hover:bg-destructive/90"
                >
                  Clear
                </Button>
              </div>
            </Card>
          </div>
        )}
      </div>

      {/* Backdrop for mobile */}
      {open && (
        <div
          className="fixed inset-0 bg-black/50 lg:hidden z-20"
          onClick={() => onOpenChange(false)}
        />
      )}
    </>
  )
}
