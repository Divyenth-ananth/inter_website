'use client'

import { useState } from 'react'
import { useTheme } from '@/components/theme-provider'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'

interface ChatHeaderProps {
  showSettings: boolean
  onSettingsChange: (show: boolean) => void
}

export default function ChatHeader({
  showSettings,
  onSettingsChange,
}: ChatHeaderProps) {
  const { theme, toggleTheme } = useTheme()

  return (
    <>
      <div className="border-b border-border bg-card px-4 lg:px-6 py-4 flex items-center justify-between">
        <h1 className="text-lg font-bold text-foreground hidden sm:block">Chat about this image</h1>
        
        <div className="flex items-center gap-2">
          <Button
            onClick={toggleTheme}
            variant="outline"
            size="sm"
            className="text-foreground hover:bg-secondary"
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </Button>
          
          <Button
            onClick={() => onSettingsChange(!showSettings)}
            variant="outline"
            size="sm"
            className="text-foreground hover:bg-secondary"
          >
            ⚙️ Settings
          </Button>
        </div>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="border-b border-border bg-card px-4 lg:px-6 py-4">
          <Card className="bg-background border-border/50 p-4">
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-foreground mb-3">Preferences</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-foreground">Theme</span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">
                        {theme === 'dark' ? 'Dark' : 'Light'}
                      </span>
                      <button
                        onClick={toggleTheme}
                        className="relative inline-flex h-6 w-11 items-center rounded-full bg-muted"
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-primary transition ${
                            theme === 'dark' ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>
                  </div>

                  <div className="pt-3 border-t border-border">
                    <p className="text-xs text-muted-foreground">
                      Font size: <span className="text-foreground">Default</span>
                    </p>
                  </div>

                  <div className="pt-3 border-t border-border">
                    <p className="text-xs text-muted-foreground">
                      About this app: v1.0.0
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </>
  )
}
