import { resumePendingJobs } from '../utils/video-job-runner'

export default defineNitroPlugin(() => {
  resumePendingJobs()
})
