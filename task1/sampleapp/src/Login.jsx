import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [errors, setErrors] = useState({})
  const [serverError, setServerError] = useState('')
  const [success, setSuccess] = useState('')
  const [loading, setLoading] = useState(false)

  const validate = () => {
    const e = {}
    if (!email) e.email = 'Email is required.'
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) e.email = 'Email is invalid.'
    if (!password) e.password = 'Password is required.'
    else if (password.length < 6) e.password = 'Password must be at least 6 characters.'
    return e
  }

  const handleSubmit = async (ev) => {
    ev.preventDefault()
    setServerError('')
    setSuccess('')
    const e = validate()
    setErrors(e)
    if (Object.keys(e).length) return
    setLoading(true)
    try {
      // call backend using Vite env var VITE_API_URL or fallback to localhost:5000
      const base = import.meta.env.VITE_API_URL || 'http://localhost:5000'
      const res = await fetch(`${base}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })
      const data = await res.json()
      if (!res.ok) setServerError(data.message || 'Login failed')
      else {
        setSuccess('Logged in successfully')
        // navigate to home page on success
        navigate('/home')
      }
    } catch (err) {
      setServerError('Network error')
    } finally {
      setLoading(false)
    }
  }

  const navigate = useNavigate()

  return (
    <form onSubmit={handleSubmit} style={{ maxWidth: 420, margin: '0 auto' }} noValidate>
      {serverError && <div className="error">{serverError}</div>}
      {success && <div className="success">{success}</div>}

      <label htmlFor="email">Email</label>
      <input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} />
      {errors.email && <div className="error">{errors.email}</div>}

      <label htmlFor="password">Password</label>
      <input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
      {errors.password && <div className="error">{errors.password}</div>}

      <button type="submit" disabled={loading}>{loading ? 'Signing in...' : 'Sign In'}</button>
    </form>
  )
}
