var spec = {
  "openapi": "3.0.0",
  "info": {
    "title": "AutoRAG API",
    "description": "API for AutoRAG with Preparation and Run workflow",
    "version": "1.0.1"
  },
  "components": {
    "schemas": {
      "Project": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "example": "proj_123"
          },
          "name": {
            "type": "string"
          },
          "description": {
            "type": "string"
          },
          "created_at": {
            "type": "string",
            "format": "date-time"
          },
          "status": {
            "type": "string",
            "enum": [
              "active",
              "archived"
            ]
          }
        }
      },
      "TrialConfig": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string"
          },
          "project_id": {
            "type": "string"
          },
          "name": {
            "type": "string"
          },
          "config_yaml": {
            "type": "string"
          },
          "created_at": {
            "type": "string",
            "format": "date-time"
          },
          "is_default": {
            "type": "boolean"
          },
          "metadata": {
            "type": "object"
          }
        }
      },
      "Task": {
        "type": "object",
        "required": [
          "id",
          "project_id",
          "status",
          "type"
        ],
        "properties": {
          "id": {
            "type": "string",
            "description": "The task id"
          },
          "project_id": {
            "type": "string"
          },
          "trial_id": {
            "type": "string"
          },
          "name": {
            "type": "string",
            "description": "The name of the task"
          },
          "config_yaml": {
            "type": "object",
            "description": "YAML configuration. Format is dictionary, not path of the YAML file."
          },
          "status": {
            "type": "string",
            "enum": [
              "not_started",
              "in_progress",
              "completed",
              "failed"
            ]
          },
          "error_message": {
            "type": "string",
            "description": "Error message if the task failed"
          },
          "type": {
            "type": "string",
            "enum": [
              "parse",
              "chunk",
              "qa",
              "validate",
              "evaluate"
            ],
            "description": "Type of the task - preparation tasks (parse, chunk, qa) or run tasks (validate, evaluate)"
          },
          "created_at": {
            "type": "string",
            "format": "date-time"
          },
          "save_path": {
            "type": "string",
            "description": "Path where the task results are saved. It will be directory or file."
          }
        }
      },
      "Trial": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string"
          },
          "project_id": {
            "type": "string"
          },
          "name": {
            "type": "string"
          },
          "preparation_id": {
            "type": "string",
            "description": "Reference to completed preparation data"
          },
          "config_yaml": {
            "type": "string",
            "description": "YAML configuration for trial"
          },
          "status": {
            "type": "string",
            "enum": [
              "not_started",
              "in_progress",
              "completed",
              "failed"
            ]
          },
          "created_at": {
            "type": "string",
            "format": "date-time"
          }
        }
      },
      "Run": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string"
          },
          "trial_id": {
            "type": "string"
          },
          "type": {
            "type": "string",
            "enum": [
              "validation",
              "eval"
            ]
          },
          "status": {
            "type": "string",
            "enum": [
              "not_started",
              "in_progress",
              "completed",
              "failed"
            ]
          },
          "result": {
            "type": "object"
          },
          "created_at": {
            "type": "string",
            "format": "date-time"
          }
        }
      }
    }
  },
  "paths": {
    "/projects": {
      "post": {
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "summary": "Create a new project",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": [
                  "name"
                ],
                "properties": {
                  "name": {
                    "type": "string"
                  },
                  "description": {
                    "type": "string"
                  }
                }
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Project created successfully",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Project"
                }
              }
            }
          },
          "400": {
            "description": "Project name already exists"
          },
          "401": {
            "description": "Unauthorized - Invalid or missing token"
          },
          "403": {
            "description": "Forbidden - Token does not have sufficient permissions"
          }
        }
      },
      "get": {
        "summary": "List all projects",
        "parameters": [
          {
            "in": "query",
            "name": "page",
            "schema": {
              "type": "integer",
              "default": 1
            }
          },
          {
            "in": "query",
            "name": "limit",
            "schema": {
              "type": "integer",
              "default": 10
            }
          },
          {
            "in": "query",
            "name": "status",
            "schema": {
              "type": "string",
              "enum": [
                "active",
                "archived"
              ]
            }
          }
        ],
        "responses": {
          "200": {
            "description": "List of projects",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "total": {
                      "type": "integer"
                    },
                    "data": {
                      "type": "array",
                      "items": {
                        "$ref": "#/components/schemas/Project"
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials": {
      "get": {
        "summary": "Get list trials",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "in": "query",
            "name": "page",
            "schema": {
              "type": "integer",
              "default": 1
            }
          },
          {
            "in": "query",
            "name": "limit",
            "schema": {
              "type": "integer",
              "default": 10
            }
          },
          {
            "in": "query",
            "name": "status",
            "schema": {
              "type": "string",
              "enum": [
                "active",
                "archived"
              ]
            }
          }
        ],
        "responses": {
          "200": {
            "description": "List of trials",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "total": {
                      "type": "integer"
                    },
                    "data": {
                      "type": "array",
                      "items": {
                        "$ref": "#/components/schemas/Trial"
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Creat trial",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": [
                  "glob_path",
                  "name"
                ],
                "properties": {
                  "glob_path": {
                    "type": "string",
                    "description": "Path pattern to match files"
                  },
                  "name": {
                    "type": "string",
                    "description": "Name for this preparation task"
                  },
                  "config": {
                    "type": "object",
                    "properties": {
                      "parse": {
                        "type": "object"
                      }
                    }
                  }
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Parsing started",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Task"
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/parse": {
      "post": {
        "summary": "Start parsing preparation",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": [
                  "glob_path",
                  "name"
                ],
                "properties": {
                  "glob_path": {
                    "type": "string",
                    "description": "Path pattern to match files"
                  },
                  "name": {
                    "type": "string",
                    "description": "Name for this preparation task"
                  },
                  "config": {
                    "type": "object",
                    "properties": {
                      "parse": {
                        "type": "object"
                      }
                    }
                  }
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Parsing started",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Task"
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/chunk": {
      "post": {
        "summary": "Start chunking preparation",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": [
                  "trial_id",
                  "name",
                  "config"
                ],
                "properties": {
                  "trial_id": {
                    "type": "string",
                    "description": "Trial ID from parsing step"
                  },
                  "name": {
                    "type": "string"
                  },
                  "config": {
                    "type": "object",
                    "properties": {
                      "chunk": {
                        "type": "object"
                      }
                    }
                  }
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Chunking started",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Task"
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/qa": {
      "post": {
        "summary": "Start QA generation preparation",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": [
                  "trial_id",
                  "name",
                  "qa_num",
                  "preset",
                  "llm_config"
                ],
                "properties": {
                  "trial_id": {
                    "type": "string",
                    "description": "Trial ID from chunking step"
                  },
                  "name": {
                    "type": "string"
                  },
                  "qa_num": {
                    "type": "integer"
                  },
                  "preset": {
                    "type": "string",
                    "enum": [
                      "basic",
                      "simple",
                      "advanced"
                    ]
                  },
                  "llm_config": {
                    "type": "object",
                    "properties": {
                      "llm_name": {
                        "type": "string"
                      },
                      "llm_params": {
                        "type": "object"
                      }
                    }
                  }
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "QA generation started",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Task"
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/config": {
      "get": {
        "summary": "Get config of trial",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "202": {
            "description": "Parsing started",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TrialConfig"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Set config of trial",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": [
                  "glob_path",
                  "name"
                ],
                "properties": {
                  "glob_path": {
                    "type": "string",
                    "description": "Path pattern to match files"
                  },
                  "name": {
                    "type": "string",
                    "description": "Name for this preparation task"
                  },
                  "config": {
                    "type": "object",
                    "properties": {
                      "parse": {
                        "type": "object"
                      }
                    }
                  }
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Parsing started",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/TrialConfig"
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/clone": {
      "post": {
        "summary": "Clone validation",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": [
                  "config",
                  "qa_path",
                  "corpus_path"
                ],
                "properties": {
                  "config": {
                    "type": "object",
                    "description": "YAML configuration. Format is dictionary, not path of the YAML file"
                  },
                  "qa_path": {
                    "type": "string",
                    "description": "Path to the QA data"
                  },
                  "corpus_path": {
                    "type": "string",
                    "description": "Path to the corpus data"
                  }
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Validation started",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Task"
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/validate": {
      "post": {
        "summary": "Start validation",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": [
                  "config",
                  "qa_path",
                  "corpus_path"
                ],
                "properties": {
                  "config": {
                    "type": "object",
                    "description": "YAML configuration. Format is dictionary, not path of the YAML file"
                  },
                  "qa_path": {
                    "type": "string",
                    "description": "Path to the QA data"
                  },
                  "corpus_path": {
                    "type": "string",
                    "description": "Path to the corpus data"
                  }
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Validation started",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Task"
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/evaluate": {
      "post": {
        "summary": "Start evaluation",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": [
                  "config_yaml",
                  "qa_path",
                  "corpus_path"
                ],
                "properties": {
                  "config_yaml": {
                    "type": "object",
                    "description": "YAML configuration. Format is dictionary, not path of the YAML file"
                  },
                  "skip_validation": {
                    "type": "boolean",
                    "description": "Skip validation step",
                    "default": true
                  },
                  "qa_path": {
                    "type": "string",
                    "description": "Path to the QA data"
                  },
                  "corpus_path": {
                    "type": "string",
                    "description": "Path to the corpus data"
                  }
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Evaluation started",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Task"
                    },
                    {
                      "type": "object",
                      "properties": {
                        "trial_id": {
                          "type": "string"
                        }
                      }
                    }
                  ]
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/report/open": {
      "get": {
        "summary": "Get preparation task or run status",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "202": {
            "description": "Evaluation started",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Task"
                    },
                    {
                      "type": "object",
                      "properties": {
                        "trial_id": {
                          "type": "string"
                        }
                      }
                    }
                  ]
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/report/close": {
      "get": {
        "summary": "Get preparation task or run status",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "202": {
            "description": "Evaluation started",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Task"
                    },
                    {
                      "type": "object",
                      "properties": {
                        "trial_id": {
                          "type": "string"
                        }
                      }
                    }
                  ]
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/chat/open": {
      "get": {
        "summary": "Get preparation task or run status",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "202": {
            "description": "Evaluation started",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Task"
                    },
                    {
                      "type": "object",
                      "properties": {
                        "trial_id": {
                          "type": "string"
                        }
                      }
                    }
                  ]
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/trials/{trial_id}/chat/close": {
      "get": {
        "summary": "Get preparation task or run status",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "trial_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "202": {
            "description": "Evaluation started",
            "content": {
              "application/json": {
                "schema": {
                  "allOf": [
                    {
                      "$ref": "#/components/schemas/Task"
                    },
                    {
                      "type": "object",
                      "properties": {
                        "trial_id": {
                          "type": "string"
                        }
                      }
                    }
                  ]
                }
              }
            }
          }
        }
      }
    },
    "/projects/{project_id}/tasks/{task_id}": {
      "get": {
        "summary": "Get preparation task or run status",
        "parameters": [
          {
            "name": "project_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "task_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Task status",
            "content": {
              "application/json": {
                "schema": {
                  "oneOf": [
                    {
                      "$ref": "#/components/schemas/Task"
                    }
                  ]
                }
              }
            }
          }
        }
      }
    }
  }
}
