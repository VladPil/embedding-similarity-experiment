import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import TextEdit from '../views/TextEdit.vue'
import BookAnalysis from '../views/BookAnalysis.vue'
import History from '../views/History.vue'
import Tasks from '../views/Tasks.vue'
import Settings from '../views/Settings.vue'
import NotFound from '../views/NotFound.vue'
import ServerError from '../views/ServerError.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home
    },
    {
      path: '/text/:id',
      name: 'text-edit',
      component: TextEdit
    },
    {
      path: '/text/:id/analyze',
      name: 'book-analysis',
      component: BookAnalysis
    },
    {
      path: '/history',
      name: 'history',
      component: History
    },
    {
      path: '/tasks',
      name: 'tasks',
      component: Tasks
    },
    {
      path: '/settings',
      name: 'settings',
      component: Settings
    },
    {
      path: '/404',
      name: 'not-found',
      component: NotFound
    },
    {
      path: '/500',
      name: 'server-error',
      component: ServerError
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/404'
    }
  ]
})

export default router
