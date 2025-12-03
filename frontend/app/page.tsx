'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import Link from 'next/link'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-background flex items-center justify-center p-4">
      {/* Background gradient accent */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-primary/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-primary/5 rounded-full blur-3xl"></div>
      </div>

      {/* Main card */}
      <Card className="w-full max-w-md bg-card border-border/50 relative z-10">
        <div className="p-8 space-y-6">
          {/* Header */}
          <div className="text-center space-y-2">
            <div className="flex items-center justify-center gap-2 mb-4">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <span className="text-primary-foreground text-lg">🛰</span>
              </div>
              <h1 className="text-xl font-bold text-foreground">Satellite Image Chat</h1>
            </div>
            <p className="text-sm text-muted-foreground">
              Unlock the power of satellite imagery with AI.
            </p>
          </div>

          {/* Sign up form */}
          <SignUpForm />

          {/* Login link */}
          <div className="text-center text-sm text-muted-foreground">
            Already have an account?{' '}
            <Link href="/login" className="text-primary hover:text-primary/80 font-medium">
              Log in
            </Link>
          </div>
        </div>
      </Card>
    </div>
  )
}

function SignUpForm() {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',
  })
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // In a real app, you'd send this to a backend
    localStorage.setItem('user', JSON.stringify({
      fullName: formData.fullName,
      email: formData.email,
    }))
    window.location.href = '/chat'
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <label className="text-sm font-medium text-foreground">Full Name</label>
        <Input
          type="text"
          name="fullName"
          placeholder="Enter your full name"
          value={formData.fullName}
          onChange={handleChange}
          required
          className="bg-background border-border text-foreground placeholder:text-muted-foreground"
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium text-foreground">Email Address</label>
        <Input
          type="email"
          name="email"
          placeholder="you@example.com"
          value={formData.email}
          onChange={handleChange}
          required
          className="bg-background border-border text-foreground placeholder:text-muted-foreground"
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium text-foreground">Password</label>
        <div className="relative">
          <Input
            type={showPassword ? 'text' : 'password'}
            name="password"
            placeholder="Enter your password"
            value={formData.password}
            onChange={handleChange}
            required
            className="bg-background border-border text-foreground placeholder:text-muted-foreground pr-10"
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          >
            {showPassword ? '👁️' : '👁️‍🗨️'}
          </button>
        </div>
        <p className="text-xs text-muted-foreground">
          Minimum 8 characters, 1 number, 1 symbol.
        </p>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium text-foreground">Confirm Password</label>
        <div className="relative">
          <Input
            type={showConfirmPassword ? 'text' : 'password'}
            name="confirmPassword"
            placeholder="Confirm your password"
            value={formData.confirmPassword}
            onChange={handleChange}
            required
            className="bg-background border-border text-foreground placeholder:text-muted-foreground pr-10"
          />
          <button
            type="button"
            onClick={() => setShowConfirmPassword(!showConfirmPassword)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          >
            {showConfirmPassword ? '👁️' : '👁️‍🗨️'}
          </button>
        </div>
      </div>

      <Button 
        type="submit" 
        className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-medium"
      >
        Create Account
      </Button>
    </form>
  )
}
