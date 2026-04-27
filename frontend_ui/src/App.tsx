import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Send, FilePlus2, Search, CheckCircle2, Star,
  ChevronDown, ChevronUp, RefreshCw, FileText,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'

/* ─── Types ──────────────────────────────────────────────────────── */
interface Citation {
  title: string
  url: string
  score: number
  snippet: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  confidence?: number
  timestamp: Date
  isLoading?: boolean
}

/* ─── Config ─────────────────────────────────────────────────────── */
const API_BASE = '/api'

const QUICK_ACTIONS = [
  {
    icon: FilePlus2,
    label: 'Déposer une réclamation',
    desc:  'Initier un nouveau dossier',
    q:     'Comment soumettre une réclamation sur le portail ?',
  },
  {
    icon: Search,
    label: 'Suivre ma réclamation',
    desc:  'Consulter l\'état de mon dossier',
    q:     'Comment suivre ma réclamation ?',
  },
  {
    icon: CheckCircle2,
    label: 'Clôturer un dossier',
    desc:  'Clôturer ou réouvrir une réclamation',
    q:     'Comment clôturer ou réouvrir une réclamation ?',
  },
  {
    icon: Star,
    label: 'Satisfaction',
    desc:  'Questionnaire de satisfaction',
    q:     'Comment accéder au questionnaire de satisfaction ?',
  },
]

const SUGGESTIONS = [
  'Comment déposer une réclamation ?',
  'Quels documents fournir ?',
  'Délai de traitement ?',
  'كيفاش نتبع ملفي ؟',
]

async function sendQuery(question: string, conversationId?: string) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, conversation_id: conversationId }),
  })
  if (!res.ok) throw new Error('network_error')
  return res.json()
}

/* ─── TypingIndicator ────────────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="flex items-center gap-[5px] py-1">
      <div className="typing-dot" />
      <div className="typing-dot" />
      <div className="typing-dot" />
    </div>
  )
}

/* ─── ConfidenceIndicator ────────────────────────────────────────── */
function ConfidenceIndicator({ score }: { score: number }) {
  const { cls, label } =
    score >= 0.70 ? { cls: 'confidence-high',   label: 'Haute confiance'     } :
    score >= 0.40 ? { cls: 'confidence-medium', label: 'Confiance modérée'   } :
                    { cls: 'confidence-low',     label: 'Information à vérifier' }
  return (
    <span className={`inline-flex items-center gap-1 text-[11px] font-medium ${cls}`}>
      <span className="w-1.5 h-1.5 rounded-full bg-current" />
      {label}
    </span>
  )
}

/* ─── CitationCard ───────────────────────────────────────────────── */
function CitationCard({ citation }: { citation: Citation }) {
  const handleClick = () => {
    if (citation.url) window.open(citation.url, '_blank', 'noopener')
  }
  return (
    <div
      className="citation-card p-3 cursor-pointer"
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && handleClick()}
      aria-label={`Source: ${citation.title}`}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-[13px] font-semibold leading-snug truncate" style={{ color: 'var(--acaps-navy)' }}>
          <FileText className="inline-block w-3 h-3 mr-1.5 opacity-60" style={{ verticalAlign: '-0.05em' }} />
          {citation.title}
        </p>
        <span className="score-pill shrink-0">{Math.round(citation.score * 100)}%</span>
      </div>
      <p className="mt-1.5 text-[12px] leading-relaxed italic line-clamp-2" style={{ color: 'var(--text-secondary)' }}>
        «&nbsp;{citation.snippet}&nbsp;»
      </p>
    </div>
  )
}

/* ─── CitationGroup ──────────────────────────────────────────────── */
function CitationGroup({ citations, confidence }: { citations: Citation[]; confidence: number }) {
  const [open, setOpen] = useState(true)
  if (!citations.length) return null
  return (
    <div className="mt-2.5 w-full animate-fade-in">
      {/* Confidence row */}
      <div className="flex items-center justify-between px-0.5 mb-1.5">
        <ConfidenceIndicator score={confidence} />
        <button
          onClick={() => setOpen(o => !o)}
          className="flex items-center gap-1 text-[11px] font-medium transition-colors hover:opacity-70"
          style={{ color: 'var(--text-muted)' }}
          aria-expanded={open}
        >
          Sources ({citations.length})
          {open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>
      </div>
      {open && (
        <div className="space-y-2">
          {citations.map((c, i) => <CitationCard key={i} citation={c} />)}
        </div>
      )}
    </div>
  )
}

/* ─── MessageBubble ──────────────────────────────────────────────── */
function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  const time   = message.timestamp.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })

  return (
    <div
      className={`flex gap-3 message-enter ${isUser ? 'flex-row-reverse' : ''}`}
      role="article"
    >
      {/* Bot avatar — hidden for user */}
      {!isUser && (
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5 font-display font-semibold text-sm select-none"
          style={{ background: 'var(--acaps-navy)', color: '#fff', minWidth: '2rem' }}
          aria-hidden="true"
        >
          A
        </div>
      )}

      {/* Content */}
      <div className={`flex flex-col gap-1 ${isUser ? 'items-end max-w-[78%]' : 'flex-1 max-w-[78%]'}`}>
        <div className={`px-4 py-3 ${isUser ? 'bubble-user' : 'bubble-bot'}`}>
          {message.isLoading ? (
            <TypingIndicator />
          ) : isUser ? (
            <div className="prose-user">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          ) : (
            <div className="prose-portal">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Citations */}
        {!isUser && !message.isLoading && message.citations && message.citations.length > 0 && (
          <CitationGroup citations={message.citations} confidence={message.confidence ?? 0} />
        )}

        {/* Timestamp */}
        <time className="text-[10px] px-0.5" style={{ color: 'var(--text-muted)' }}>
          {time}
        </time>
      </div>
    </div>
  )
}

/* ─── QuickActions ───────────────────────────────────────────────── */
function QuickActions({ onSelect }: { onSelect: (q: string) => void }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 w-full max-w-chat mx-auto mt-2">
      {QUICK_ACTIONS.map((a) => {
        const Icon = a.icon
        return (
          <button
            key={a.label}
            className="qa-card p-4 text-left flex flex-col gap-3"
            onClick={() => onSelect(a.q)}
            aria-label={a.label}
          >
            <div
              className="w-11 h-11 rounded-xl flex items-center justify-center shrink-0"
              style={{ background: 'rgba(0,163,224,0.08)' }}
            >
              <Icon className="w-5 h-5" style={{ color: 'var(--acaps-sky)', strokeWidth: 2 }} />
            </div>
            <div>
              <p className="text-[14px] font-semibold leading-snug" style={{ color: 'var(--acaps-navy)' }}>
                {a.label}
              </p>
              <p className="text-[12px] mt-0.5 hidden sm:block" style={{ color: 'var(--text-secondary)' }}>
                {a.desc}
              </p>
            </div>
          </button>
        )
      })}
    </div>
  )
}

/* ─── WelcomeScreen ──────────────────────────────────────────────── */
function WelcomeScreen({ onSelect }: { onSelect: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center text-center pt-8 pb-4 px-4 animate-fade-in">
      {/* Logo */}
      <img
        src="/acaps-logo.jpg"
        alt="ACAPS"
        className="h-14 w-auto object-contain mb-6"
      />

      {/* Headline */}
      <h2
        className="font-display text-3xl font-500 leading-snug mb-2"
        style={{ color: 'var(--acaps-navy)', fontFamily: 'Fraunces, Georgia, serif', fontWeight: 500 }}
      >
        Bonjour, je suis Atlas.
      </h2>
      <p className="text-base leading-relaxed mb-1" style={{ color: 'var(--text-secondary)', maxWidth: '480px' }}>
        Votre assistant pour le portail des réclamations ACAPS.
      </p>
      <p className="text-sm mb-8" style={{ color: 'var(--text-muted)' }}>
        Posez vos questions en français, en arabe ou en darija&nbsp;—&nbsp;
        <span style={{ fontFamily: "'IBM Plex Sans Arabic', sans-serif" }}>أنا هنا.</span>
      </p>

      {/* Quick actions */}
      <QuickActions onSelect={onSelect} />

      {/* Suggestion chips */}
      <div className="flex flex-wrap justify-center gap-2 mt-8 max-w-lg">
        <p className="w-full text-xs font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
          Suggestions populaires :
        </p>
        {SUGGESTIONS.map((s, i) => (
          <button
            key={i}
            className="suggestion-chip px-4 py-2 text-sm font-medium"
            style={{ color: 'var(--acaps-navy)' }}
            onClick={() => onSelect(s)}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}

/* ─── ChatInput ──────────────────────────────────────────────────── */
function ChatInput({
  value, onChange, onSubmit, disabled,
}: {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  disabled: boolean
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-grow textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'
  }, [value])

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSubmit()
    }
  }

  return (
    <div>
      <div className="flex items-end gap-3">
        <textarea
          ref={textareaRef}
          className="chat-input flex-1 px-4 py-3"
          style={{ minHeight: '48px', maxHeight: '160px' }}
          placeholder="Posez votre question… / اطرح سؤالك…"
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={handleKey}
          disabled={disabled}
          rows={1}
          aria-label="Zone de saisie"
          autoComplete="off"
          spellCheck
        />
        <button
          className="send-btn w-11 h-11 flex items-center justify-center shrink-0 text-white"
          style={{ minWidth: '44px' }}
          onClick={onSubmit}
          disabled={!value.trim() || disabled}
          aria-label="Envoyer le message"
        >
          <Send className="w-[18px] h-[18px]" style={{ strokeWidth: 2.25 }} />
        </button>
      </div>
      <p className="mt-2 text-[11px] text-center" style={{ color: 'var(--text-muted)' }}>
        L'assistant peut se tromper. Vérifiez les sources citées.
      </p>
    </div>
  )
}

/* ─── Header ─────────────────────────────────────────────────────── */
function Header({ onReset }: { onReset: () => void }) {
  return (
    <header
      className="shrink-0 bg-white border-b px-5 sm:px-8"
      style={{
        borderColor: 'var(--border-subtle)',
        boxShadow: '0 1px 4px rgba(26,58,92,0.06)',
        height: '64px',
        display: 'flex',
        alignItems: 'center',
      }}
    >
      <div className="flex items-center justify-between w-full max-w-chat mx-auto">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <img src="/acaps-logo.jpg" alt="ACAPS" className="h-8 w-auto object-contain" />
          <div className="hidden sm:block">
            <p className="text-[15px] font-semibold leading-tight" style={{ color: 'var(--acaps-navy)' }}>
              Guide Portail
            </p>
            <p className="text-[11px] font-medium flex items-center gap-1.5" style={{ color: 'var(--success)' }}>
              <span className="w-1.5 h-1.5 rounded-full bg-current status-pulse" />
              Assistant en ligne
            </p>
          </div>
        </div>

        {/* Reset */}
        <button
          onClick={onReset}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors hover:bg-gray-50"
          style={{ color: 'var(--text-secondary)' }}
          aria-label="Nouvelle conversation"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Nouvelle conversation</span>
        </button>
      </div>
    </header>
  )
}

/* ─── App ────────────────────────────────────────────────────────── */
export default function App() {
  const [messages, setMessages]     = useState<Message[]>([])
  const [input, setInput]           = useState('')
  const [isLoading, setIsLoading]   = useState(false)
  const [conversationId, setConvId] = useState<string | undefined>()
  const [error, setError]           = useState<string | null>(null)
  const [showWelcome, setShowWelcome] = useState(true)

  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const submit = useCallback(async (question: string) => {
    if (!question.trim() || isLoading) return
    setShowWelcome(false)
    setError(null)

    const userMsg: Message = {
      id:        `u${Date.now()}`,
      role:      'user',
      content:   question,
      timestamp: new Date(),
    }
    const loadingMsg: Message = {
      id:        `a${Date.now() + 1}`,
      role:      'assistant',
      content:   '',
      timestamp: new Date(),
      isLoading: true,
    }

    setMessages(prev => [...prev, userMsg, loadingMsg])
    setInput('')
    setIsLoading(true)

    try {
      const res = await sendQuery(question, conversationId)
      setConvId(res.conversation_id)
      setMessages(prev =>
        prev.map(m =>
          m.id === loadingMsg.id
            ? {
                ...loadingMsg,
                content:    res.answer,
                citations:  res.citations ?? [],
                confidence: res.confidence ?? 0,
                isLoading:  false,
              }
            : m
        )
      )
    } catch {
      setError('Une erreur est survenue. Veuillez réessayer.')
      setMessages(prev => prev.filter(m => m.id !== loadingMsg.id))
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, conversationId])

  const handleReset = () => {
    setMessages([])
    setConvId(undefined)
    setError(null)
    setShowWelcome(true)
    setInput('')
  }

  return (
    <div className="h-screen flex flex-col dot-grid" style={{ background: 'var(--bg-page)' }}>

      <Header onReset={handleReset} />

      {/* ── Messages / Welcome ── */}
      <main
        className="flex-1 overflow-y-auto"
        role="log"
        aria-live="polite"
        aria-atomic="false"
        aria-label="Conversation"
      >
        <div className="max-w-chat mx-auto px-4 sm:px-6 py-6 space-y-6">
          {showWelcome && messages.length === 0 ? (
            <WelcomeScreen onSelect={q => submit(q)} />
          ) : (
            messages.map(m => <MessageBubble key={m.id} message={m} />)
          )}
          <div ref={endRef} />
        </div>
      </main>

      {/* ── Error bar ── */}
      {error && (
        <div
          className="shrink-0 px-6 py-2.5 border-t text-sm text-center"
          style={{ background: '#FEF2F2', borderColor: '#FECACA', color: 'var(--error)' }}
          role="alert"
        >
          {error}
        </div>
      )}

      {/* ── Input ── */}
      <footer
        className="shrink-0 bg-white border-t px-5 sm:px-8 pt-4 pb-5"
        style={{
          borderColor: 'var(--border-subtle)',
          boxShadow: '0 -1px 8px rgba(26,58,92,0.05)',
          paddingBottom: 'max(20px, env(safe-area-inset-bottom))',
        }}
      >
        <div className="max-w-chat mx-auto">
          <ChatInput
            value={input}
            onChange={setInput}
            onSubmit={() => submit(input.trim())}
            disabled={isLoading}
          />
        </div>
      </footer>

    </div>
  )
}
