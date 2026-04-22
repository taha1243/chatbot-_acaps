/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        acaps: {
          sky:       '#00A3E0',
          'sky-600': '#0088BD',
          'sky-50':  '#E6F6FC',
          navy:      '#1A3A5C',
          'navy-700':'#234A74',
          sand:      '#B5A89A',
          'sand-50': '#F7F4F0',
          bg:        '#FAFBFC',
          surface:   '#FFFFFF',
          muted:     '#F4F6F8',
          text:      '#0D1F3C',
          secondary: '#5A6A7E',
          subtle:    '#8A94A3',
          border:    '#E4E8EC',
        },
      },
      fontFamily: {
        sans:    ['"Inter"', 'system-ui', 'sans-serif'],
        display: ['"Fraunces"', 'Georgia', 'serif'],
        arabic:  ['"IBM Plex Sans Arabic"', 'system-ui', 'sans-serif'],
      },
      maxWidth: { chat: '820px' },
      animation: {
        'slide-up':   'slideUp 240ms cubic-bezier(0.2, 0.8, 0.2, 1)',
        'fade-in':    'fadeIn 200ms ease-out',
        'typing':     'typingBounce 1.2s ease-in-out infinite',
        'pulse-ring': 'pulseRing 2s ease-in-out infinite',
      },
      keyframes: {
        slideUp: {
          '0%':   { transform: 'translateY(12px)', opacity: '0' },
          '100%': { transform: 'translateY(0)',    opacity: '1' },
        },
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        typingBounce: {
          '0%, 60%, 100%': { transform: 'translateY(0)',    opacity: '0.4' },
          '30%':           { transform: 'translateY(-4px)', opacity: '1'   },
        },
        pulseRing: {
          '0%, 100%': { opacity: '1',   transform: 'scale(1)' },
          '50%':      { opacity: '0.5', transform: 'scale(0.85)' },
        },
      },
    },
  },
  plugins: [],
}
