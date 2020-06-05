#!/usr/bin/env groovy
// Adapted from github.com/dmlc/dgl/Jenkinsfile

app = "dgllife"

def init_git() {
  sh "rm -rf *"
  checkout scm
  sh "git submodule update --recursive --init"
}

def build_linux(dev) {
  init_git()
  sh "bash tests/scripts/build.sh ${dev}"
}

def unit_test_linux(backend, dev) {
  timeout(time: 10, unit: 'MINUTES') {
    sh "bash tests/scripts/task_unit_test.sh ${backend} ${dev}"
  }
}

pipeline {
  agent any
  stages {
    stage("Lint Check") {
      agent {
        docker {
          label "linux-c52x-node"
          image "dgllib/dgl-ci-lint"
        }
      }
      steps {
        init_git()
        sh "bash tests/scripts/task_lint.sh"
      }
      post {
        always {
          cleanWs disableDeferredWipeout: true, deleteDirs: true
        }
      }
    }
    stage("Build") {
      parallel {
        stage("CPU Build") {
          agent {
            docker {
              label "linux-c52x-node"
              image "dgllib/${app}-ci-cpu"
              alwaysPull true
            }
          }
          steps {
            build_linux("cpu")
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("GPU Build") {
          agent {
            docker {
              label "linux-c52x-node"
              image "dgllib/${app}-ci-gpu:latest"
              args "-u root"
              alwaysPull true
            }
          }
          steps {
            build_linux("gpu")
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
      }
    }
    stage("Test") {
      parallel {
        stage("Torch CPU") {
          agent {
            docker {
              label "linux-c52x-node"
              image "dgllib/${app}-ci-cpu:latest"
            }
          }
          stages {
            stage("Unit test") {
              steps {
                unit_test_linux("pytorch", "cpu")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
        stage("Torch GPU") {
          agent {
            docker {
              label "linux-gpu-node"
              image "dgllib/${app}-ci-gpu:latest"
              args "--runtime nvidia"
            }
          }
          stages {
            stage("Unit test") {
              steps {
                sh "nvidia-smi"
                unit_test_linux("pytorch", "gpu")
              }
            }
          }
          post {
            always {
              cleanWs disableDeferredWipeout: true, deleteDirs: true
            }
          }
        }
      }
    }
  }
}
