

import React from 'react'
import './App.css'
import { Routes, Route } from 'react-router-dom'
import Login from './Login'
import Home from './Home'

function App() {
  return (
    <main id="center">
      <Routes>
        <Route path="/" element={<>
          <div>
            <h1>Welcome</h1>
            <p>Sign in to continue</p>
          </div>
          <Login />
        </>} />
        <Route path="/home" element={<Home />} />
      </Routes>
    </main>
  )
}

export default App
