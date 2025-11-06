import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import Home from '@/views/Home.vue'
import { useTextsStore } from '@/store/texts'
import { useAnalysisStore } from '@/store/analysis'
import { useSettingsStore } from '@/store/settings'

// Mock router
vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: vi.fn()
  }),
  useRoute: () => ({
    path: '/'
  })
}))

describe('Home.vue', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders correctly', () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    expect(wrapper.find('.page-layout').exists()).toBe(true)
    expect(wrapper.find('.main-content').exists()).toBe(true)
  })

  it('shows upload section when button clicked', async () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    // Initially hidden
    expect(wrapper.find('.upload-section').exists()).toBe(false)

    // Click upload button
    const uploadBtn = wrapper.find('button:contains("Добавить текст")')
    await uploadBtn.trigger('click')

    // Should show upload section
    expect(wrapper.find('.upload-section').exists()).toBe(true)
  })

  it('displays texts from store', async () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    const textsStore = useTextsStore()

    // Add test texts
    textsStore.texts = [
      { id: '1', title: 'Test Text 1', content: 'Content 1', length: 100, lines: 5 },
      { id: '2', title: 'Test Text 2', content: 'Content 2', length: 200, lines: 10 }
    ]

    await wrapper.vm.$nextTick()

    // Check texts are displayed
    const textCards = wrapper.findAll('.text-card')
    expect(textCards.length).toBe(2)
    expect(textCards[0].text()).toContain('Test Text 1')
    expect(textCards[1].text()).toContain('Test Text 2')
  })

  it('enables analysis when 2 texts selected', async () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    const textsStore = useTextsStore()
    textsStore.texts = [
      { id: '1', title: 'Text 1', content: 'Content 1', length: 100, lines: 5 },
      { id: '2', title: 'Text 2', content: 'Content 2', length: 200, lines: 10 }
    ]

    await wrapper.vm.$nextTick()

    // Select first text
    const selectBtns = wrapper.findAll('.btn-select')
    await selectBtns[0].trigger('click')

    // Analysis section should not show yet
    expect(wrapper.find('.analysis-section').exists()).toBe(false)

    // Select second text
    await selectBtns[1].trigger('click')

    // Analysis section should show
    expect(wrapper.find('.analysis-section').exists()).toBe(true)
    expect(wrapper.find('.analysis-methods').exists()).toBe(true)
  })

  it('runs semantic analysis', async () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    const analysisStore = useAnalysisStore()
    const analyzeSpy = vi.spyOn(analysisStore, 'analyze').mockResolvedValue({
      similarity: 0.85,
      interpretation: 'High similarity'
    })

    // Setup selected texts
    wrapper.vm.selectedTexts = [
      { id: '1', title: 'Text 1' },
      { id: '2', title: 'Text 2' }
    ]

    await wrapper.vm.$nextTick()

    // Click semantic analysis button
    const semanticBtn = wrapper.find('button:contains("Семантика")')
    await semanticBtn.trigger('click')

    expect(analyzeSpy).toHaveBeenCalledWith(
      'semantic',
      expect.objectContaining({
        text_id1: '1',
        text_id2: '2'
      }),
      'Text 1',
      'Text 2'
    )
  })

  it('displays analysis results with LLM metrics', async () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    const analysisStore = useAnalysisStore()
    analysisStore.results = [{
      type: 'llm',
      similarity: 0.75,
      plot_similarity: 0.8,
      style_similarity: 0.7,
      genre_similarity: 0.85,
      characters_similarity: 0.6,
      language_similarity: 0.9,
      interpretation: 'LLM analysis complete'
    }]

    await wrapper.vm.$nextTick()

    // Check results section exists
    expect(wrapper.find('.results-section').exists()).toBe(true)

    // Check LLM metrics are displayed
    const metricsSection = wrapper.find('.llm-metrics')
    expect(metricsSection.exists()).toBe(true)

    const metricItems = wrapper.findAll('.metric-item')
    expect(metricItems.length).toBeGreaterThan(0)
  })

  it('displays combined analysis breakdown', async () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    const analysisStore = useAnalysisStore()
    analysisStore.results = [{
      type: 'combined',
      similarity: 0.72,
      results: {
        semantic: { similarity: 0.8, weight: 0.5, weighted_score: 0.4 },
        style: { similarity: 0.6, weight: 0.3, weighted_score: 0.18 },
        tfidf: { similarity: 0.7, weight: 0.2, weighted_score: 0.14 }
      },
      interpretation: 'Combined analysis complete'
    }]

    await wrapper.vm.$nextTick()

    // Check combined breakdown exists
    const breakdown = wrapper.find('.combined-breakdown')
    expect(breakdown.exists()).toBe(true)

    const breakdownItems = wrapper.findAll('.breakdown-item')
    expect(breakdownItems.length).toBe(3)
  })

  it('handles text upload', async () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    const textsStore = useTextsStore()
    const uploadSpy = vi.spyOn(textsStore, 'uploadText').mockResolvedValue()

    // Show upload form
    wrapper.vm.showUpload = true
    wrapper.vm.uploadTitle = 'New Text'
    wrapper.vm.uploadText = 'New content'

    await wrapper.vm.$nextTick()

    // Submit upload
    const uploadBtn = wrapper.find('button:contains("Загрузить")')
    await uploadBtn.trigger('click')

    expect(uploadSpy).toHaveBeenCalledWith('New Text', 'New content')
  })

  it('handles text deletion', async () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    const textsStore = useTextsStore()
    const deleteSpy = vi.spyOn(textsStore, 'deleteText').mockResolvedValue()

    textsStore.texts = [
      { id: '1', title: 'Text to delete', content: 'Content', length: 100, lines: 5 }
    ]

    wrapper.vm.selectedForDelete = ['1']
    await wrapper.vm.$nextTick()

    // Delete selected
    const deleteBtn = wrapper.find('.btn-danger:contains("Удалить")')
    await deleteBtn.trigger('click')

    expect(deleteSpy).toHaveBeenCalledWith('1')
  })

  it('formats numbers correctly', () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    // Test formatNumber function
    expect(wrapper.vm.formatNumber(1000)).toBe('1,000')
    expect(wrapper.vm.formatNumber(1000000)).toBe('1,000,000')
  })

  it('calculates score colors correctly', () => {
    const wrapper = mount(Home, {
      global: {
        stubs: {
          AppHeader: true,
          HistorySidebar: true,
          Toast: true
        }
      }
    })

    // Test getScoreColor function
    expect(wrapper.vm.getScoreColor(1)).toBe('hsl(120, 70%, 50%)')  // Green
    expect(wrapper.vm.getScoreColor(0.5)).toBe('hsl(60, 70%, 50%)')  // Yellow
    expect(wrapper.vm.getScoreColor(0)).toBe('hsl(0, 70%, 50%)')    // Red
  })
})