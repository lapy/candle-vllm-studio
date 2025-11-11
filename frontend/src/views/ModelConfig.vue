<template>
  <div class="model-config">
    <div class="config-layout">
      <!-- Main Configuration Panel -->
      <div class="config-main">
        <div class="card">
          <div class="card-header">
            <div class="config-header">
              <div class="model-info">
                <h2 class="card-title">{{ model?.huggingface_id || model?.name || 'Model Configuration' }}</h2>
                <div class="model-tags">
                  <span class="model-tag tag-size">{{ formatFileSize(model?.file_size) }}</span>
                  <span class="model-tag tag-quantization">{{ model?.quantization }}</span>
                  <span class="model-tag tag-type">{{ model?.model_type }}</span>
                  <span v-if="modelLayerInfo?.architecture && modelLayerInfo.architecture !== model?.model_type"
                    class="model-tag tag-architecture">
                    {{ modelLayerInfo.architecture }}
                  </span>
                  <span v-if="modelLayerInfo?.layer_count" class="model-tag tag-layers">
                    {{ modelLayerInfo.layer_count }} layers
                  </span>
                </div>
              </div>
              <div class="header-actions">
                <div class="action-buttons">
                  <Button 
                    label="Quick Start" 
                    icon="pi pi-bolt" 
                    size="small"
                    severity="info"
                    outlined
                    @click="showQuickStartModal = true"
                    v-tooltip="'Choose a preset, use the wizard, or let Smart Auto optimize for you'"
                  />
                  <Button icon="pi pi-refresh" @click="regenerateModelInfo" :loading="regeneratingInfo"
                    severity="secondary" size="small" outlined
                    v-tooltip="'Regenerate model information from GGUF metadata'" />
                  <Button label="Save Config" icon="pi pi-save" @click="saveConfig" :loading="saveLoading"
                    severity="success" size="small" />
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Quick Start Modal -->
        <Dialog 
          v-model:visible="showQuickStartModal" 
          modal 
          :closable="true"
          :dismissableMask="true"
          :draggable="false"
          class="quick-start-modal"
          @hide="showQuickStartModal = false"
        >
          <template #header>
            <div class="quick-start-modal-header">
              <div class="quick-start-icon">🚀</div>
              <div>
                <h3>Quick Start</h3>
                <p>Choose a preset, use the wizard, or let Smart Auto optimize for you</p>
              </div>
            </div>
          </template>

          <div class="quick-start-content">
            <div class="preset-cards">
              <div 
                class="preset-card wizard-card" 
                role="button"
                tabindex="0"
                aria-label="Configuration Wizard - Guided 3-step setup for new users"
                @click="showWizard = true; showQuickStartModal = false"
                @keydown.enter="showWizard = true; showQuickStartModal = false"
                @keydown.space.prevent="showWizard = true; showQuickStartModal = false"
              >
                <div class="preset-icon" aria-hidden="true">✨</div>
                <div class="preset-info">
                  <h4>Configuration Wizard</h4>
                  <p>Guided 3-step setup for new users</p>
                </div>
              </div>
              <div 
                class="preset-card" 
                role="button"
                tabindex="0"
                aria-label="Coding preset - Low temperature, high precision for code generation"
                @click="applyPreset('coding'); showQuickStartModal = false"
                @keydown.enter="applyPreset('coding'); showQuickStartModal = false"
                @keydown.space.prevent="applyPreset('coding'); showQuickStartModal = false"
              >
                <div class="preset-icon" aria-hidden="true">💻</div>
                <div class="preset-info">
                  <h4>Coding</h4>
                  <p>Low temperature, high precision for code generation</p>
                </div>
              </div>
              <div 
                class="preset-card" 
                role="button"
                tabindex="0"
                aria-label="Chat preset - Balanced settings for natural conversation"
                @click="applyPreset('conversational'); showQuickStartModal = false"
                @keydown.enter="applyPreset('conversational'); showQuickStartModal = false"
                @keydown.space.prevent="applyPreset('conversational'); showQuickStartModal = false"
              >
                <div class="preset-icon" aria-hidden="true">💬</div>
                <div class="preset-info">
                  <h4>Chat</h4>
                  <p>Balanced settings for natural conversation</p>
                </div>
              </div>
            </div>
            
            <div class="smart-auto-section">
              <div class="smart-auto-header">
                <i class="pi pi-bolt"></i>
                <h4>Smart Auto Configuration</h4>
              </div>
              <p class="smart-auto-description">Automatically optimize settings based on your hardware and use case</p>
              
              <div class="usage-mode-selector" role="radiogroup" aria-label="Usage mode selection">
                <div 
                  class="radio-option" 
                  role="radio"
                  :aria-checked="smartAutoUsageMode === 'single_user'"
                  tabindex="0"
                  aria-label="Single User mode - Sequential requests, maximum context"
                  @click="smartAutoUsageMode = 'single_user'"
                  @keydown.enter="smartAutoUsageMode = 'single_user'"
                  @keydown.space.prevent="smartAutoUsageMode = 'single_user'"
                  :class="{ active: smartAutoUsageMode === 'single_user' }"
                >
                  <i class="pi pi-user" aria-hidden="true"></i>
                  <div>
                    <strong>Single User</strong>
                    <small>Sequential requests, maximum context</small>
                  </div>
                </div>
                <div 
                  class="radio-option" 
                  role="radio"
                  :aria-checked="smartAutoUsageMode === 'multi_user'"
                  tabindex="0"
                  aria-label="Multi User Server mode - Parallel requests, optimized batching"
                  @click="smartAutoUsageMode = 'multi_user'"
                  @keydown.enter="smartAutoUsageMode = 'multi_user'"
                  @keydown.space.prevent="smartAutoUsageMode = 'multi_user'"
                  :class="{ active: smartAutoUsageMode === 'multi_user' }"
                >
                  <i class="pi pi-users" aria-hidden="true"></i>
                  <div>
                    <strong>Multi User Server</strong>
                    <small>Parallel requests, optimized batching</small>
                  </div>
                </div>
              </div>
              <div class="smart-auto-controls">
                <div class="control-row">
                  <div class="control-label">
                    <label for="smart-auto-speed">Speed vs Quality</label>
                    <span class="control-hint">{{ smartAutoSpeedDescription }}</span>
                  </div>
                  <Slider
                    id="smart-auto-speed"
                    v-model="smartAutoSpeedQuality"
                    :min="0"
                    :max="100"
                    :step="1"
                    aria-label="Speed quality balance slider"
                  />
                </div>

                <div class="control-row">
                  <div class="control-label">
                    <label for="smart-auto-usecase">Use Case</label>
                    <span class="control-hint">{{ smartAutoUseCaseDescription }}</span>
                  </div>
                  <Dropdown
                    id="smart-auto-usecase"
                    v-model="smartAutoUseCase"
                    :options="smartAutoUseCaseOptions"
                    optionLabel="label"
                    optionValue="value"
                    placeholder="Select use case"
                    class="smart-auto-usecase"
                  />
                </div>

                <div class="control-grid">
                  <div class="control-field">
                    <label for="smart-auto-concurrency">
                      Target Concurrency
                      <small>(optional)</small>
                    </label>
                    <InputNumber
                      id="smart-auto-concurrency"
                      v-model="smartAutoTargetConcurrency"
                      :min="1"
                      :max="16"
                      :useGrouping="false"
                      placeholder="Auto"
                    />
                  </div>
                  <div class="control-field">
                    <label for="smart-auto-max-tokens">
                      Max Tokens Hint
                      <small>(optional)</small>
                    </label>
                    <InputNumber
                      id="smart-auto-max-tokens"
                      v-model="smartAutoMaxTokensHint"
                      :min="256"
                      :step="64"
                      :useGrouping="false"
                      placeholder="Auto"
                    />
                  </div>
                </div>

                <div class="control-row debug-toggle">
                  <Checkbox v-model="smartAutoIncludeDebug" inputId="smart-auto-debug" binary />
                  <label for="smart-auto-debug">Include decision rationale in preview</label>
                </div>
                <div class="control-row debug-toggle">
                  <Checkbox v-model="smartAutoRestrictNvlink" inputId="smart-auto-nvlink" binary />
                  <label for="smart-auto-nvlink">Restrict to NVLink groups when available</label>
                </div>
              </div>

              <Button 
                label="Generate Optimal Config" 
                icon="pi pi-bolt" 
                @click="generateAutoConfig" 
                :loading="autoConfigLoading" 
                size="large" 
                class="smart-auto-button"
                aria-label="Generate optimal configuration automatically based on hardware and use case"
              />
            </div>
          </div>
          
          <template #footer>
            <Button label="Close" icon="pi pi-times" @click="showQuickStartModal = false" severity="secondary" />
          </template>
        </Dialog>

        <!-- Empty State for New Models -->
        <EmptyState
          v-if="hasNoConfig && !showWizard && !showPreview"
          :visible="hasNoConfig"
          icon="🎯"
          title="Configure Your Model"
          description="This model doesn't have a configuration yet. Start with Smart Auto, choose a preset, or configure manually."
          :show-smart-auto="true"
          :show-presets="true"
          @smart-auto="generateAutoConfig"
          @presets="() => activeTabIndex = 0"
          @manual="() => activeTabIndex = 0"
        />

        <!-- Configuration Grid -->
        <div 
          v-if="!hasNoConfig || showWizard || showPreview" 
          class="config-tabs-wrapper"
          @touchstart="handleTabSwipeStart"
          @touchend="handleTabSwipeEnd"
        >
          <div class="config-search-bar" :class="{ 'search-focused': searchFocused }">
            <span class="p-input-icon-left">
              <i class="pi pi-search" />
              <InputText 
                v-model="configSearchQuery" 
                placeholder="Search settings..."
                class="config-search-input"
                aria-label="Search configuration settings"
                @focus="searchFocused = true"
                @blur="searchFocused = false"
              />
            </span>
            <Button 
              v-if="configSearchQuery"
              icon="pi pi-times" 
              @click="configSearchQuery = ''" 
              size="small" 
              text 
              rounded
              severity="secondary"
              aria-label="Clear search"
              v-tooltip="'Clear search'"
            />
          </div>
          <div class="info-banner">
            <i class="pi pi-info-circle"></i>
            <div class="banner-content">
              <strong>About Candle-vLLM Configuration</strong>
              <p>Candle-vLLM starts an OpenAI-compatible server. Generation parameters (temperature, top-k, sampling, etc.) are configured per-request via the API, not at startup. This UI configures the runtime server settings only.</p>
            </div>
          </div>
          <div v-if="topologyPlan?.global" class="topology-summary-card">
            <div class="topology-summary-header">
              <div>
                <h4>Topology Plan</h4>
                <p>
                  {{ topologyPlan.global.tp_strategy === 'tensor_parallel' ? 'Tensor Parallel' : 'Replica Execution' }}
                  ·
                  {{ topologyPlan.global.mixed_topology ? 'Mixed NVLink/PCIe' : 'Homogeneous Topology' }}
                  <template v-if="topologyPlan.global.restrict_to_nvlink">
                    · NVLink-only shards
                  </template>
                </p>
              </div>
              <div class="topology-summary-metric">
                <span class="metric-label">KV Pool</span>
                <span class="metric-value">{{ formatFileSize(topologyPlan.global.total_kv_pool_mb * 1024 * 1024) }}</span>
              </div>
            </div>
            <div class="topology-summary-grid">
              <div>
                <span class="metric-label">Recommended Concurrency</span>
                <span class="metric-value">{{ topologyPlan.global.recommended_max_num_seqs }}</span>
              </div>
              <div>
                <span class="metric-label">Prefill Chunk Size</span>
                <span class="metric-value">{{ topologyPlan.global.prefill_chunk_size }}</span>
              </div>
              <div>
                <span class="metric-label">Block Size</span>
                <span class="metric-value">{{ topologyPlan.global.block_size }}</span>
              </div>
              <div>
                <span class="metric-label">FP8 KV Cache</span>
                <span class="metric-value">{{ topologyPlan.global.fp8_enabled ? 'Enabled' : 'Disabled' }}</span>
              </div>
            </div>
            <div v-if="topologyPlan.global.warnings?.length" class="topology-warnings">
              <i class="pi pi-exclamation-triangle"></i>
              <ul>
                <li v-for="(warning, idx) in topologyPlan.global.warnings" :key="idx">{{ warning }}</li>
              </ul>
            </div>
            <div class="topology-groups">
              <div
                v-for="(group, groupIndex) in topologyPlan.groups"
                :key="group.id || groupIndex"
                class="topology-group-card"
              >
                <div class="group-header">
                  <div>
                    <h5>Group {{ groupIndex + 1 }} · {{ group.link_type === 'nvlink' ? 'NVLink' : 'PCIe' }}</h5>
                    <p>TP Degree: {{ group.tp_degree }} · KV Pool: {{ formatFileSize(group.kv_pool_mb * 1024 * 1024) }}</p>
                  </div>
                  <div v-if="group.expected_penalty_pct" class="group-penalty">
                    Penalty ~{{ group.expected_penalty_pct }}%
                  </div>
                </div>
                <table class="group-device-table">
                  <thead>
                    <tr>
                      <th>GPU</th>
                      <th>Weights</th>
                      <th>Activations</th>
                      <th>KV Cache</th>
                      <th>Safety Margin</th>
                      <th>FP8</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="device in group.devices" :key="device.index">
                      <td>GPU{{ device.index }}</td>
                      <td>{{ formatFileSize(device.weights_mb * 1024 * 1024) }}</td>
                      <td>{{ formatFileSize(device.activations_mb * 1024 * 1024) }}</td>
                      <td>{{ formatFileSize(device.kv_budget_mb * 1024 * 1024) }}</td>
                      <td>{{ formatFileSize(device.safety_margin_mb * 1024 * 1024) }}</td>
                      <td>{{ device.supports_fp8 ? 'Yes' : 'No' }}</td>
                    </tr>
                  </tbody>
                </table>
                <div v-if="group.warnings?.length" class="topology-warnings group-warnings">
                  <i class="pi pi-exclamation-circle"></i>
                  <ul>
                    <li v-for="(warning, idx) in group.warnings" :key="idx">{{ warning }}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
          <TabView v-model:activeIndex="activeTabIndex" :scrollable="true" class="config-tabs">
          <!-- Essential Settings Tab -->
          <TabPanel header="Server" icon="pi pi-server">
            <div class="tab-content">
              <div class="tab-section">
                <h4 class="tab-section-title">
                  <i class="pi pi-microchip"></i>
                  Runtime Configuration
                </h4>
                <ConfigField label="Candle Build" help-text="Override the system-wide active build (leave empty to use system default).">
                  <template #input>
                    <Dropdown
                      v-model="config.build_name"
                      :options="buildOptions"
                      optionLabel="label"
                      optionValue="value"
                      placeholder="Use system default build"
                      :loading="buildsStore.loading"
                      :disabled="!buildOptions.length"
                      showClear
                    />
                  </template>
                  <template #validation>
                    <div v-if="!buildOptions.length" class="inline-validation warning">
                      <i class="pi pi-info-circle"></i>
                      <span>No builds available. Create or activate a build first.</span>
                    </div>
                  </template>
                </ConfigField>
                <ConfigField label="Weights Directory" help-text="Managed by the application (set automatically on download).">
                  <template #input>
                    <InputText v-model="config.weights_path" placeholder="e.g. /app/data/models/my-model" readonly disabled />
                  </template>
                </ConfigField>
                <ConfigField label="Weights File" help-text="GGUF file used when launching candle-vllm (read-only).">
                  <template #input>
                    <InputText v-model="config.weights_file" placeholder="e.g. /app/data/models/my-model/model.gguf" readonly disabled />
                  </template>
                </ConfigField>
                <ConfigField label="Host" help-text="Server bind address (default: 0.0.0.0)">
                  <template #input>
                    <InputText v-model="config.host" placeholder="0.0.0.0" />
                  </template>
                </ConfigField>
                <ConfigField label="Port" help-text="Server port (candle default range 3000-3999)">
                  <template #input>
                    <InputNumber
                      v-model="config.port"
                      :min="CANDLE_PORT_MIN"
                      :max="CANDLE_PORT_MAX"
                      :useGrouping="false"
                    />
                  </template>
                </ConfigField>
                <ConfigField label="Timeout (seconds)" help-text="Health check timeout for model startup">
                  <template #input>
                    <InputNumber v-model="config.timeout" :min="30" :max="600" :useGrouping="false" />
                  </template>
                </ConfigField>
              </div>
              <div class="tab-section">
                <h4 class="tab-section-title">
                  <i class="pi pi-sliders-h"></i>
                  Model Parameters
                </h4>
                <ConfigField label="KV Cache GPU Memory (MB)" help-text="GPU memory for KV cache (--mem parameter, e.g. 4096 = 4 GB)">
                  <template #input>
                    <InputNumber v-model="config.kvcache_mem_gpu" :min="0" :max="262144" :step="128" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="KV Cache CPU Memory (MB)" help-text="CPU fallback memory for KV cache (--kvcache-mem-cpu)">
                  <template #input>
                    <InputNumber v-model="config.kvcache_mem_cpu" :min="0" :max="1048576" :step="256" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Runtime DType" help-text="Model data type (--dtype parameter)">
                  <template #input>
                    <Dropdown v-model="config.dtype" :options="dtypeOptions" optionLabel="label" optionValue="value" placeholder="Auto" showClear />
                  </template>
                </ConfigField>
                <ConfigField label="In-Situ Quantization" help-text="Runtime quantization (--isq parameter)">
                  <template #input>
                    <Dropdown v-model="config.isq" :options="isqOptions" optionLabel="label" optionValue="value" placeholder="Disabled" showClear />
                  </template>
                </ConfigField>
                <ConfigField label="Max Generated Tokens" help-text="Maximum tokens per generation (--max-gen-tokens parameter)">
                  <template #input>
                    <InputNumber v-model="config.max_gen_tokens" :min="16" :max="65536" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Max Concurrent Sequences" help-text="Maximum simultaneous sequences (--max-num-seqs, default 16)">
                  <template #input>
                    <InputNumber v-model="config.max_num_seqs" :min="1" :max="4096" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="KV Block Size" help-text="KV cache block size (--block-size, default 64)">
                  <template #input>
                    <InputNumber v-model="config.block_size" :min="16" :max="1024" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Device ID" help-text="GPU device index (--d parameter, default: 0)">
                  <template #input>
                    <InputNumber v-model="config.device_id" :min="0" :max="15" :useGrouping="false" />
                  </template>
                </ConfigField>
              </div>
            </div>
          </TabPanel>

          <!-- Advanced Tab -->
          <TabPanel header="Advanced" icon="pi pi-cog">
            <div class="tab-content">
              <div class="tab-section">
                <h4 class="tab-section-title">
                  <i class="pi pi-sliders-v"></i>
                  Scheduler & Runtime Flags
                </h4>
                <ConfigField label="Force CPU Mode" help-text="Run entirely on CPU (--cpu)">
                  <template #input>
                    <Checkbox v-model="config.cpu" binary />
                  </template>
                </ConfigField>
                <ConfigField label="Record Conversations" help-text="Persist conversation transcripts (--record-conversation)">
                  <template #input>
                    <Checkbox v-model="config.record_conversation" binary />
                  </template>
                </ConfigField>
                <ConfigField label="Verbose Logging" help-text="Verbose request logging (--verbose)">
                  <template #input>
                    <Checkbox v-model="config.verbose" binary />
                  </template>
                </ConfigField>
                <ConfigField label="Structured Log Output" help-text="Print structured logs (--log)">
                  <template #input>
                    <Checkbox v-model="config.log" binary />
                  </template>
                </ConfigField>
                <ConfigField label="Multithreaded Scheduler" help-text="Enable multithreaded scheduling (--multithread)">
                  <template #input>
                    <Checkbox v-model="config.multithread" binary />
                  </template>
                </ConfigField>
                <ConfigField label="Holding Time (ms)" help-text="Max wait time to batch requests (--holding-time)">
                  <template #input>
                    <InputNumber v-model="config.holding_time" :min="0" :max="10000" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Prefill Chunk Size" help-text="Chunk size for prefill stage (--prefill-chunk-size)">
                  <template #input>
                    <InputNumber v-model="config.prefill_chunk_size" :min="0" :max="16384" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Enable FP8 KV Cache" help-text="Use FP8 cache format (--fp8-kvcache)">
                  <template #input>
                    <Checkbox v-model="config.fp8_kvcache" binary />
                  </template>
                </ConfigField>
              </div>
              <div class="tab-section">
                <h4 class="tab-section-title">
                  <i class="pi pi-chart-line"></i>
                  Sampling Defaults
                </h4>
                <ConfigField label="Temperature" help-text="Default sampling temperature (--temperature)">
                  <template #input>
                    <InputNumber v-model="config.temperature" :min="0" :max="5" :step="0.05" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Top-P" help-text="Default nucleus sampling value (--top-p)">
                  <template #input>
                    <InputNumber v-model="config.top_p" :min="0" :max="1" :step="0.01" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Min-P" help-text="Minimum probability mass (--min-p)">
                  <template #input>
                    <InputNumber v-model="config.min_p" :min="0" :max="1" :step="0.01" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Top-K" help-text="Limit sampling to top-k tokens (--top-k)">
                  <template #input>
                    <InputNumber v-model="config.top_k" :min="0" :max="1000" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Frequency Penalty" help-text="Penalty for repeated tokens (--frequency-penalty)">
                  <template #input>
                    <InputNumber v-model="config.frequency_penalty" :min="-2" :max="2" :step="0.05" :useGrouping="false" />
                  </template>
                </ConfigField>
                <ConfigField label="Presence Penalty" help-text="Penalty for token presence (--presence-penalty)">
                  <template #input>
                    <InputNumber v-model="config.presence_penalty" :min="-2" :max="2" :step="0.05" :useGrouping="false" />
                  </template>
                </ConfigField>
              </div>
              <div class="tab-section">
                <h4 class="tab-section-title">
                  <i class="pi pi-lock"></i>
                  Authentication
                </h4>
                <ConfigField label="Hugging Face Token" help-text="Token passed via --hf-token">
                  <template #input>
                    <InputText v-model="config.hf_token" type="password" placeholder="Optional access token" />
                  </template>
                </ConfigField>
                <ConfigField label="Hugging Face Token Path" help-text="Token file path (--hf-token-path)">
                  <template #input>
                    <InputText v-model="config.hf_token_path" placeholder="e.g. /app/token.txt" />
                  </template>
                </ConfigField>
              </div>
              <div class="tab-section">
                <h4 class="tab-section-title">
                  <i class="pi pi-cog"></i>
                  Advanced Runtime Options
                </h4>
            <ConfigField label="Cargo Features" help-text="Additional Cargo features for 'cargo run' (comma-separated, e.g. cuda,flash-attn)">
              <template #input>
                <InputText v-model="featuresInput" placeholder="e.g. cuda,flash-attn,nccl" @blur="applyFeatures" />
              </template>
            </ConfigField>
            <ConfigField label="Extra CLI Arguments" help-text="Additional command-line arguments passed to candle-vllm" full-width>
              <template #input>
                <Textarea v-model="extraArgsInput" rows="3" placeholder="--arg1&#10;--arg2 value" @blur="applyExtraArgs" />
              </template>
            </ConfigField>
            <ConfigField label="Environment Variables" help-text="Custom environment variables (KEY=VALUE format, one per line)" full-width>
              <template #input>
                <Textarea v-model="envInput" rows="3" placeholder="RUST_LOG=debug&#10;CUSTOM_VAR=value" @blur="applyEnv" />
              </template>
            </ConfigField>
            <ConfigField label="Build Profile" help-text="Cargo build profile for 'cargo run' mode">
              <template #input>
                <Dropdown 
                  v-model="config.build_profile" 
                  :options="[{ label: 'Release', value: 'release' }, { label: 'Debug', value: 'debug' }]" 
                  optionLabel="label" 
                  optionValue="value" 
                />
              </template>
            </ConfigField>
            </div>
          </div>
          </TabPanel>
        </TabView>
      </div>
      </div>

      <!-- Removed Tabs Notice (for legacy configs) -->
      <div class="info-message" v-if="hasLegacyFields" style="margin-top: 1rem;">
        <i class="pi pi-info-circle"></i>
        <span>This model has legacy configuration fields that are not supported by candle-vllm. These fields are ignored at runtime.</span>
      </div>

      <!-- Configuration Warnings -->
      <div v-if="configWarnings.length > 0" class="config-warnings">
        <h3 class="section-title">
          <i class="pi pi-exclamation-triangle"></i>
          Configuration Warnings
        </h3>
        <div class="warnings-list">
          <div v-for="warning in configWarnings" :key="warning.field" class="warning-item" :class="warning.type">
            <i :class="warning.type === 'error' ? 'pi pi-times-circle' : 'pi pi-exclamation-triangle'"></i>
            <span>{{ warning.message }}</span>
          </div>
        </div>
      </div>
      
      <!-- Memory Status Dashboard (prominent top cards) -->
      <div class="memory-dashboard" id="memory-dashboard">
        <!-- RAM Monitor Card -->
        <MemoryMonitor
          title="System RAM"
          :current-value="realtimeRamData?.used || null"
          :estimated-value="ramEstimate?.estimated_ram || null"
          :total-capacity="totalRamBytes"
          :loading="ramLoading && !ramEstimate && !realtimeRamData"
        />

        <!-- VRAM Monitor Card (only shown when GPU is available) -->
        <MemoryMonitor
          v-if="!systemStore.gpuInfo.cpu_only_mode && systemStore.gpuInfo.device_count > 0"
          title="VRAM"
          :current-value="currentVramBytes"
          :estimated-value="vramEstimate?.estimated_vram || null"
          :total-capacity="totalVramBytes"
          :loading="vramLoading && !vramEstimate && !realtimeVramData"
        />
      </div>

      <!-- RAM Monitor Sidebar - DEPRECATED: Use Memory Dashboard above instead -->
      <!-- This sidebar is kept for backward compatibility but will be removed in future versions -->
      <div class="config-sidebar" style="display: none;">
        <div class="ram-monitor">
          <div class="monitor-header">
            <h3>
              <i class="pi pi-calculator"></i>
              Memory Estimation
            </h3>
            <div class="monitor-meta" v-if="ramEstimate || realtimeRamData">
              <span v-if="ramEstimate?.system_ram_total" class="ram-snapshot">
                System RAM: {{ formatFileSize(ramEstimate.system_ram_used) }} / {{
                  formatFileSize(ramEstimate.system_ram_total) }}
              </span>
              <span v-else-if="realtimeRamData" class="ram-snapshot">
                System RAM: {{ formatFileSize(realtimeRamData.used) }} / {{ formatFileSize(realtimeRamData.total) }}
              </span>
            </div>
          </div>

          <div class="monitor-content" v-if="ramEstimate || realtimeRamData">
            <template v-if="ramEstimate">
              <div class="ram-summary">
                <div class="ram-total">
                  <span class="total-label">Estimated Usage</span>
                  <span class="total-value" :class="ramEstimate.fits_in_ram ? 'success' : 'warning'">
                    {{ formatFileSize(ramEstimate.estimated_ram) }}
                  </span>
                </div>
                <div class="ram-progress">
                  <div class="stacked-bar" :class="ramEstimate.fits_in_ram ? 'success' : 'warning'">
                    <div class="bar-current" :style="{ width: currentRamPercent + '%' }"></div>
                    <div class="bar-additional"
                      :style="{ width: additionalRamPercent + '%', left: currentRamPercent + '%' }"></div>
                  </div>
                  <span class="progress-text">
                    {{ currentRamPercent }}% used + {{ additionalRamPercent }}% est • {{
                      formatFileSize(totalEstimatedRamBytes)
                    }}
                    total est
                  </span>
                </div>
              </div>

              <div class="ram-breakdown">
                <div class="breakdown-item">
                  <span class="item-label">Model</span>
                  <span class="item-value">{{ formatFileSize(ramEstimate.model_ram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">KV Cache</span>
                  <span class="item-value">{{ formatFileSize(ramEstimate.kv_cache_ram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Batch</span>
                  <span class="item-value">{{ formatFileSize(ramEstimate.batch_ram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Overhead</span>
                  <span class="item-value">{{ formatFileSize(ramEstimate.overhead_ram) }}</span>
                </div>
              </div>

              <div v-if="showRamWarning" class="ram-warning">
                <i class="pi pi-exclamation-triangle"></i>
                <span>Estimated RAM usage exceeds available system memory</span>
              </div>
            </template>
            <template v-else>
              <div class="ram-summary">
                <div class="ram-total">
                  <span class="total-label">Current System Usage</span>
                  <span class="total-value" :class="realtimeRamData.percent > 80 ? 'warning' : 'success'">
                    {{ formatFileSize(realtimeRamData.used) }}
                  </span>
                </div>
                <div class="ram-progress">
                  <ProgressBar :value="realtimeRamData.percent" :showValue="false"
                    :class="realtimeRamData.percent > 80 ? 'warning' : 'success'" />
                  <span class="progress-text">{{ (realtimeRamData.percent || 0).toFixed(1) }}% of {{
                    formatFileSize(realtimeRamData.total) }}</span>
                </div>
              </div>

              <div class="ram-breakdown">
                <div class="breakdown-item">
                  <span class="item-label">Used</span>
                  <span class="item-value">{{ formatFileSize(realtimeRamData.used) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Available</span>
                  <span class="item-value">{{ formatFileSize(realtimeRamData.available) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Cached</span>
                  <span class="item-value">{{ formatFileSize(realtimeRamData.cached || 0) }}</span>
                </div>
                <div v-if="realtimeRamData.swap_total > 0" class="breakdown-item">
                  <span class="item-label">Swap Used</span>
                  <span class="item-value">{{ formatFileSize(realtimeRamData.swap_used) }} ({{
                    (realtimeRamData.swap_percent || 0).toFixed(1) }}%)</span>
                </div>
              </div>

              <div v-if="realtimeRamData.percent > 90" class="ram-warning">
                <i class="pi pi-exclamation-triangle"></i>
                <span>High RAM usage detected</span>
              </div>
            </template>
          </div>
          <div v-else class="monitor-loading">
            <i class="pi pi-spin pi-spinner"></i>
            <span>Connecting to real-time monitoring...</span>
          </div>
        </div>

        <!-- VRAM Monitor (only shown when GPU is available) -->
        <div v-if="!systemStore.gpuInfo.cpu_only_mode && systemStore.gpuInfo.device_count > 0" class="vram-monitor">
          <div class="monitor-header">
            <h3>
              <i class="pi pi-microchip"></i>
              VRAM Monitor
            </h3>
            <div class="monitor-meta" v-if="vramEstimate">
              <span class="mode-badge">Mode: {{ vramEstimate.memory_mode || 'unknown' }}</span>
              <span v-if="vramEstimate.system_ram_total" class="ram-snapshot">
                System RAM: {{ formatFileSize(vramEstimate.system_ram_used) }} / {{
                  formatFileSize(vramEstimate.system_ram_total) }}
              </span>
            </div>
          </div>

          <div class="monitor-content" v-if="realtimeVramData || vramEstimate">
            <template v-if="realtimeVramData">
              <div class="vram-summary">
                <div class="vram-total">
                  <span class="total-label">Real-time Usage</span>
                  <span class="total-value" :class="realtimeVramData.percent > 80 ? 'warning' : 'success'">
                    {{ formatFileSize(realtimeVramData.used_vram) }}
                  </span>
                </div>
                <div class="vram-progress">
                  <div class="stacked-bar"
                    :class="(currentVramPercent + additionalVramPercent) > 80 ? 'warning' : 'success'">
                    <div class="bar-current" :style="{ width: currentVramPercent + '%' }"></div>
                    <div class="bar-additional"
                      :style="{ width: additionalVramPercent + '%', left: currentVramPercent + '%' }"></div>
                  </div>
                  <span class="progress-text">
                    {{ currentVramPercent }}% used + {{ additionalVramPercent }}% est • {{
                      formatFileSize(totalEstimatedVramBytes)
                    }} total est
                  </span>
                </div>
              </div>

              <div class="vram-breakdown">
                <div v-for="gpu in realtimeVramData.gpus" :key="gpu.index" class="gpu-item">
                  <div class="gpu-header">
                    <span class="gpu-name">{{ gpu.name }}</span>
                    <span class="gpu-temp" v-if="gpu.temperature">{{ gpu.temperature }}°C</span>
                  </div>
                  <div class="gpu-memory">
                    <span class="memory-label">VRAM</span>
                    <span class="memory-value">{{ formatFileSize(gpu.memory.used) }} / {{
                      formatFileSize(gpu.memory.total)
                    }}</span>
                    <span class="memory-percent">{{ (gpu.memory.percent || 0).toFixed(1) }}%</span>
                  </div>
                  <div v-if="gpu.utilization" class="gpu-utilization">
                    <span class="util-label">GPU</span>
                    <span class="util-value">{{ gpu.utilization.gpu || 0 }}%</span>
                    <span class="util-label">Mem</span>
                    <span class="util-value">{{ gpu.utilization.memory || 0 }}%</span>
                  </div>
                </div>
              </div>

              <div v-if="realtimeVramData.percent > 90" class="vram-warning">
                <i class="pi pi-exclamation-triangle"></i>
                <span>High VRAM usage detected</span>
              </div>
            </template>
            <template v-else>
              <div class="vram-summary">
                <div class="vram-total">
                  <span class="total-label">Estimated Usage</span>
                  <span class="total-value" :class="vramEstimate.fits_in_gpu ? 'success' : 'warning'">
                    {{ formatFileSize(vramEstimate.estimated_vram) }}
                  </span>
                </div>
                <div class="vram-progress">
                  <ProgressBar :value="vramUsagePercentage" :showValue="false"
                    :class="vramEstimate.fits_in_gpu ? 'success' : 'warning'" />
                  <span class="progress-text">{{ vramUsagePercentage }}% of available VRAM</span>
                </div>
              </div>

              <div class="vram-breakdown">
                <div class="breakdown-item">
                  <span class="item-label">Model</span>
                  <span class="item-value">{{ formatFileSize(vramEstimate.model_vram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">KV Cache</span>
                  <span class="item-value">{{ formatFileSize(vramEstimate.kv_cache_vram) }}</span>
                </div>
                <div class="breakdown-item">
                  <span class="item-label">Batch</span>
                  <span class="item-value">{{ formatFileSize(vramEstimate.batch_vram) }}</span>
                </div>
              </div>

              <div v-if="!vramEstimate.fits_in_gpu" class="vram-warning">
                <i class="pi pi-exclamation-triangle"></i>
                <span>VRAM usage exceeds available GPU memory</span>
              </div>

              <div v-if="systemStore.gpuInfo.nvlink_topology?.has_nvlink" class="nvlink-info">
                <i class="pi pi-link"></i>
                <span>{{ systemStore.gpuInfo.nvlink_topology.recommended_strategy }}</span>
              </div>
            </template>
          </div>

          <div v-else class="monitor-loading">
            <i class="pi pi-spin pi-spinner"></i>
            <span>Connecting to real-time monitoring...</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Configuration Wizard -->
    <ConfigWizard
      :model-visible="showWizard"
      :model="model"
      :gpu-info="systemStore.gpuInfo"
      :model-layer-info="modelLayerInfo"
      @close="showWizard = false"
      @apply-config="handleWizardConfig"
      @go-to-advanced="showWizard = false"
    />

    <!-- Configuration Change Preview -->
    <ConfigChangePreview
      :visible="showPreview"
      @update:visible="showPreview = $event"
      :type="previewData?.type || 'smart-auto'"
      :preset-name="previewData?.presetName || ''"
      :changes="previewData?.changes || []"
      :impact="previewData?.impact"
      :plan="previewData?.plan"
      :debug="previewData?.debug"
      :applying="previewApplying"
      @apply="applyPreviewChanges"
      @cancel="showPreview = false"
    />

    <!-- Onboarding Tour -->
    <OnboardingTour
      :visible="showOnboarding"
      :steps="onboardingSteps"
      @update:visible="showOnboarding = $event"
      @complete="handleOnboardingComplete"
      @skip="handleOnboardingSkip"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useModelStore } from '@/stores/models'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { useBuildStore } from '@/stores/builds'
import { toast } from 'vue3-toastify'
import Button from 'primevue/button'
import InputNumber from 'primevue/inputnumber'
import Slider from 'primevue/slider'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import Checkbox from 'primevue/checkbox'
import Dropdown from 'primevue/dropdown'
import ProgressBar from 'primevue/progressbar'
import TabView from 'primevue/tabview'
import TabPanel from 'primevue/tabpanel'
import Dialog from 'primevue/dialog'
import SliderInput from '@/components/SliderInput.vue'
import MemoryMonitor from '@/components/config/MemoryMonitor.vue'
import ConfigSection from '@/components/config/ConfigSection.vue'
import ConfigField from '@/components/config/ConfigField.vue'
import ConfigWizard from '@/components/config/ConfigWizard.vue'
import ConfigChangePreview from '@/components/config/ConfigChangePreview.vue'
import OnboardingTour from '@/components/config/OnboardingTour.vue'

const route = useRoute()
const router = useRouter()
const modelStore = useModelStore()
const systemStore = useSystemStore()
const wsStore = useWebSocketStore()
const buildsStore = useBuildStore()

// Reactive state
const model = ref(null)
const config = ref({})
const featuresInput = ref('')
const extraArgsInput = ref('')
const envInput = ref('')

const dtypeOptions = [
  { label: 'Auto', value: null },
  { label: 'FP16', value: 'f16' },
  { label: 'BF16', value: 'bf16' },
  { label: 'FP32', value: 'f32' }
]

const isqOptions = [
  { label: 'Disabled', value: null },
  { label: 'Q2_K', value: 'q2k' },
  { label: 'Q3_K', value: 'q3k' },
  { label: 'Q4_K', value: 'q4k' },
  { label: 'Q5_K', value: 'q5k' },
  { label: 'Q6_K', value: 'q6k' },
  { label: 'Q4_0', value: 'q4_0' },
  { label: 'Q4_1', value: 'q4_1' },
  { label: 'Q5_0', value: 'q5_0' },
  { label: 'Q5_1', value: 'q5_1' },
  { label: 'Q8_0', value: 'q8_0' },
  { label: 'AWQ', value: 'awq' },
  { label: 'GPTQ', value: 'gptq' },
  { label: 'Marlin', value: 'marlin' },
  { label: 'GGUF', value: 'gguf' },
  { label: 'GGML', value: 'ggml' }
]

const smartAutoUseCaseOptions = [
  { label: 'Auto Detect', value: null, description: 'Let Smart Auto choose based on model metadata.' },
  { label: 'Conversational', value: 'chat', description: 'Balanced dialogue for assistants and chatbots.' },
  { label: 'Code', value: 'code', description: 'Stable, low-temperature responses for code generation.' },
  { label: 'Analysis', value: 'analysis', description: 'Long-form reasoning, summarisation and planning.' },
  { label: 'Creative', value: 'creative', description: 'Higher diversity for brainstorming and storytelling.' }
]


const CANDLE_PORT_MIN = 3000
const CANDLE_PORT_MAX = 3999
const DEFAULT_CANDLE_PORT = 3000

const buildOptions = computed(() =>
  (buildsStore.builds || []).map((build) => ({
    label: build.name,
    value: build.name
  }))
)

const normalizePathString = (path) => {
  if (!path) return ''
  return String(path).replace(/\\/g, '/')
}

const deriveWeightsDirectory = () => {
  const filePath = model.value?.file_path
  if (!filePath) return ''
  const normalized = normalizePathString(filePath)
  if (normalized.toLowerCase().endsWith('.gguf') || normalized.toLowerCase().endsWith('.ggml')) {
    const idx = normalized.lastIndexOf('/')
    return idx > 0 ? normalized.slice(0, idx) : ''
  }
  return normalized
}

const normalizeRuntimeConfig = () => {
  if (!config.value) return

  if (!config.value.host) {
    config.value.host = '0.0.0.0'
  }

  const parsedPort = Number(config.value.port)
  if (!Number.isFinite(parsedPort)) {
    config.value.port = DEFAULT_CANDLE_PORT
  } else {
    const rounded = Math.round(parsedPort)
    if (rounded < CANDLE_PORT_MIN || rounded > CANDLE_PORT_MAX) {
      config.value.port = DEFAULT_CANDLE_PORT
    } else {
      config.value.port = rounded
    }
  }

  let weightsPath = config.value.weights_path
  if (!weightsPath || String(weightsPath).trim() === '') {
    weightsPath = deriveWeightsDirectory()
  }
  weightsPath = normalizePathString(weightsPath)
  if (weightsPath.toLowerCase().endsWith('.gguf') || weightsPath.toLowerCase().endsWith('.ggml')) {
    const idx = weightsPath.lastIndexOf('/')
    weightsPath = idx > 0 ? weightsPath.slice(0, idx) : ''
  }
  config.value.weights_path = weightsPath

  let weightsFile = config.value.weights_file
  if (!weightsFile || String(weightsFile).trim() === '') {
    const modelFile = model.value?.file_path
    weightsFile = modelFile ? normalizePathString(modelFile) : ''
  } else {
    weightsFile = normalizePathString(weightsFile)
    if (!weightsFile.toLowerCase().endsWith('.gguf') && weightsPath && weightsFile && !weightsFile.includes('/')) {
      const base = weightsPath.endsWith('/') ? weightsPath.slice(0, -1) : weightsPath
      weightsFile = `${base}/${weightsFile}`
    }
  }
  config.value.weights_file = weightsFile

  let cacheValue = Number(config.value.kvcache_mem_gpu)
  if (!Number.isFinite(cacheValue) || cacheValue <= 0) {
    cacheValue = 4096
  } else if (cacheValue <= 64) {
    cacheValue = Math.round(cacheValue * 1024)
  } else {
    cacheValue = Math.round(cacheValue)
  }
  cacheValue = Math.min(Math.max(cacheValue, 0), 262144)
  config.value.kvcache_mem_gpu = cacheValue

  let cpuCacheValue = Number(config.value.kvcache_mem_cpu)
  if (!Number.isFinite(cpuCacheValue) || cpuCacheValue <= 0) {
    config.value.kvcache_mem_cpu = null
  } else if (cpuCacheValue <= 64) {
    config.value.kvcache_mem_cpu = Math.min(Math.round(cpuCacheValue * 1024), 1048576)
  } else {
    config.value.kvcache_mem_cpu = Math.min(Math.round(cpuCacheValue), 1048576)
  }

  if (config.value.dtype === '') {
    config.value.dtype = null
  }
  if (config.value.isq === '') {
    config.value.isq = null
  }
  if (!Array.isArray(config.value.features)) {
    config.value.features = []
  }
  if (!Array.isArray(config.value.extra_args)) {
    config.value.extra_args = []
  }
  if (typeof config.value.env !== 'object' || config.value.env === null) {
    config.value.env = {}
  }
  const intFields = ['max_num_seqs', 'block_size', 'prefill_chunk_size', 'holding_time', 'top_k', 'device_id']
  intFields.forEach((field) => {
    const value = config.value[field]
    if (value === '' || value === null || value === undefined) {
      config.value[field] = null
      return
    }
    const parsed = Number(value)
    config.value[field] = Number.isFinite(parsed) ? Math.round(parsed) : null
    if (config.value[field] !== null && config.value[field] < 0) {
      config.value[field] = null
    }
  })

  const floatFields = [
    { key: 'temperature', min: 0, max: 5 },
    { key: 'top_p', min: 0, max: 1 },
    { key: 'min_p', min: 0, max: 1 },
    { key: 'frequency_penalty', min: -2, max: 2 },
    { key: 'presence_penalty', min: -2, max: 2 }
  ]
  floatFields.forEach(({ key, min, max }) => {
    const value = config.value[key]
    if (value === '' || value === null || value === undefined) {
      config.value[key] = null
      return
    }
    const parsed = Number(value)
    if (!Number.isFinite(parsed)) {
      config.value[key] = null
      return
    }
    const clamped = Math.min(Math.max(parsed, min), max)
    config.value[key] = Number.isNaN(clamped) ? null : clamped
  })

  const boolFields = ['verbose', 'cpu', 'record_conversation', 'multithread', 'log', 'fp8_kvcache']
  boolFields.forEach((field) => {
    config.value[field] = Boolean(config.value[field])
  })

  if (typeof config.value.hf_token !== 'string') {
    config.value.hf_token = config.value.hf_token ?? ''
  }
  if (typeof config.value.hf_token_path !== 'string') {
    config.value.hf_token_path = config.value.hf_token_path ?? ''
  }

  if (config.value.device_id === null) {
    config.value.device_id = 0
  }
  if (!config.value.build_name && buildOptions.value.length) {
    const active = (buildsStore.builds || []).find((build) => build.is_active)
    config.value.build_name = active ? active.name : buildOptions.value[0].value
  }
}

watch(
  () => buildsStore.builds,
  (builds) => {
    if (!config.value || !Array.isArray(builds) || !builds.length) return
    if (!config.value.build_name) {
      const active = builds.find((build) => build.is_active)
      config.value.build_name = active ? active.name : builds[0].name
    }
  },
  { immediate: true }
)

watch(
  () => model.value?.file_path,
  () => {
    normalizeRuntimeConfig()
  }
)
const applyStopWords = () => {
  const parts = (stopWordsInput.value || '').split(',').map(s => s.trim()).filter(Boolean)
  config.value.stop = parts
}
const applyFeatures = () => {
  const parts = (featuresInput.value || '').split(',').map(s => s.trim()).filter(Boolean)
  config.value.features = parts
}
const applyExtraArgs = () => {
  const lines = (extraArgsInput.value || '').split('\n').map(s => s.trim()).filter(Boolean)
  config.value.extra_args = lines
}
const applyEnv = () => {
  const lines = (envInput.value || '').split('\n').map(s => s.trim()).filter(Boolean)
  const env = {}
  lines.forEach(line => {
    const match = line.match(/^([^=]+)=(.*)$/)
    if (match) {
      env[match[1].trim()] = match[2].trim()
    }
  })
  config.value.env = env
}
const vramEstimate = ref(null)
const vramLoading = ref(false)
const ramEstimate = ref(null)
const ramLoading = ref(false)
let vramEstimateTimeout = null
let ramEstimateTimeout = null
let vramAbortController = null
let ramAbortController = null
const ESTIMATE_DEBOUNCE_MS = 400
const autoConfigLoading = ref(false)
const saveLoading = ref(false)
const modelLayerInfo = ref(null)
const modelRecommendations = ref(null)
const regeneratingInfo = ref(false)
const layerInfoLoading = ref(false)
const recommendationsLoading = ref(false)
const smartAutoUsageMode = ref('single_user')
const smartAutoSpeedQuality = ref(50)
const smartAutoUseCase = ref(null)
const smartAutoTargetConcurrency = ref(null)
const smartAutoMaxTokensHint = ref(null)
const smartAutoIncludeDebug = ref(false)
const smartAutoDebug = ref(null)
const smartAutoRestrictNvlink = ref(false)
const topologyPlan = ref(null)

const smartAutoSpeedDescription = computed(() => {
  const value = smartAutoSpeedQuality.value
  if (value <= 15) return 'Max speed'
  if (value <= 40) return 'Speed leaning'
  if (value <= 60) return 'Balanced'
  if (value <= 85) return 'Quality leaning'
  return 'Max quality'
})

const smartAutoUseCaseDescription = computed(() => {
  const match = smartAutoUseCaseOptions.find(option => option.value === smartAutoUseCase.value)
  return match?.description || ''
})

const toOptionalInt = (value) => {
  if (value === null || value === undefined || value === '') return null
  const parsed = Number(value)
  return Number.isNaN(parsed) ? null : parsed
}
const configSearchQuery = ref('')
const searchFocused = ref(false)
const showWizard = ref(false)
const showQuickStartModal = ref(false)

// Real-time memory data from WebSocket
const realtimeRamData = ref(null)
const realtimeVramData = ref(null)

const normalizeGpuMemoryEntry = (gpu, index) => {
  if (!gpu) return null
  const idx = gpu.index ?? gpu.device_id ?? index
  const total = gpu.memory?.total ?? gpu.total ?? 0
  const used = gpu.memory?.used ?? gpu.used ?? 0
  const free = gpu.memory?.free ?? gpu.free ?? Math.max(total - used, 0)
  const percent = gpu.memory?.percent ?? (total ? (used / total) * 100 : 0)

  return {
    index: idx,
    name: gpu.name ?? gpu.model ?? `GPU ${idx}`,
    memory: {
      total,
      used,
      free,
      percent,
    },
    utilization: {
      gpu: gpu.utilization?.gpu ?? gpu.utilization ?? 0,
      memory: gpu.utilization?.memory ?? gpu.memory_utilization ?? 0,
    },
    temperature: gpu.temperature ?? null,
    raw: gpu,
  }
}

const normalizeVramData = (vramData) => {
  if (!vramData) return null
  const gpusArray = Array.isArray(vramData.gpus) ? vramData.gpus : []
  const gpus = gpusArray
    .map((gpu, idx) => normalizeGpuMemoryEntry(gpu, idx))
    .filter(Boolean)

  const totalFromGpus = gpus.reduce((sum, gpu) => sum + (gpu.memory?.total ?? 0), 0)
  const usedFromGpus = gpus.reduce((sum, gpu) => sum + (gpu.memory?.used ?? 0), 0)

  const total = vramData.total ?? vramData.total_vram ?? totalFromGpus
  const used = vramData.used ?? vramData.used_vram ?? usedFromGpus
  const free = vramData.free ?? vramData.free_vram ?? Math.max(total - used, 0)
  const percent = vramData.percent ?? (total ? (used / total) * 100 : 0)

  return {
    ...vramData,
    total,
    used_vram: used,
    free_vram: free,
    percent,
    gpus,
  }
}

// Performance metrics from unified monitoring

// Active tab index for tabbed interface
const activeTabIndex = ref(0)

// Tab labels for search matching
const tabLabels = [
  { key: 'essential', label: 'Essential', keywords: ['gpu', 'layer', 'thread', 'cpu', 'loading', 'model'] },
  { key: 'memory', label: 'Memory & Context', keywords: ['context', 'memory', 'batch', 'size', 'mmap', 'mlock', 'ram'] },
  { key: 'generation', label: 'Generation', keywords: ['temperature', 'temp', 'top-k', 'top-p', 'repeat', 'penalty', 'sampling', 'token', 'generate', 'mirostat', 'seed', 'grammar', 'jinja'] },
  { key: 'performance', label: 'Performance', keywords: ['parallel', 'flash', 'attention', 'vram', 'low', 'batching', 'offload', 'logits', 'embedding', 'cache', 'kv', 'quantization', 'moe', 'expert'] },
  { key: 'advanced', label: 'Advanced', keywords: ['rope', 'yarn', 'freq', 'scale', 'scaling', 'yaml'] },
  { key: 'custom', label: 'Custom Arguments', keywords: ['custom', 'argument', 'args', 'command', 'line'] }
]

// Swipe between tabs functionality
const tabSwipeStartX = ref(0)
const tabSwipeThreshold = 100

const handleTabSwipeStart = (e) => {
  if (!e.touches || e.touches.length === 0) return
  tabSwipeStartX.value = e.touches[0].clientX
}

const handleTabSwipeEnd = (e) => {
  if (!e.changedTouches || e.changedTouches.length === 0) return
  
  const deltaX = e.changedTouches[0].clientX - tabSwipeStartX.value
  
  // Swipe left (next tab) or right (previous tab)
  if (Math.abs(deltaX) > tabSwipeThreshold) {
    if (deltaX < 0 && activeTabIndex.value < tabLabels.length - 1) {
      // Swipe left - next tab
      activeTabIndex.value++
    } else if (deltaX > 0 && activeTabIndex.value > 0) {
      // Swipe right - previous tab
      activeTabIndex.value--
    }
  }
  
  tabSwipeStartX.value = 0
}

// Supported runtime flags (currently placeholder until candle exposes CLI metadata)
const supportedFlags = ref(null)
const supportedConfigKeys = ref({})

// GPU options
const gpuOptions = computed(() => {
  return Array.from({ length: systemStore.gpuInfo.device_count }, (_, i) => ({
    label: `GPU ${i}`,
    value: i
  }))
})

// GPU availability
const gpuAvailable = computed(() => {
  const dc = systemStore.gpuInfo?.device_count || 0
  return dc > 0
})

watch(gpuAvailable, (avail) => {
  if (!avail) {
    if (config.value && config.value.n_gpu_layers !== 0) {
      config.value.n_gpu_layers = 0
      updateVramEstimate(true)
    }
  }
})

onMounted(() => {
  if (!gpuAvailable.value) {
    if (config.value && config.value.n_gpu_layers !== 0) {
      config.value.n_gpu_layers = 0
      updateVramEstimate(true)
    }
  }
})

// VRAM usage percentage
const vramUsagePercentage = computed(() => {
  if (!vramEstimate.value || !systemStore.gpuInfo.total_vram) return 0
  return Math.round((vramEstimate.value.estimated_vram / systemStore.gpuInfo.total_vram) * 100)
})

// RAM usage percentage
const ramUsagePercentage = computed(() => {
  if (!ramEstimate.value || !systemStore.systemStatus.system?.memory?.total) return 0
  return Math.round((ramEstimate.value.estimated_ram / systemStore.systemStatus.system.memory.total) * 100)
})

// Only show RAM warning when we have valid totals and estimate exceeds available
const showRamWarning = computed(() => {
  const est = ramEstimate.value
  if (!est) return false
  const total = est.system_ram_total || systemStore.systemStatus.system?.memory?.total || 0
  const estimated = est.estimated_ram || 0
  return total > 0 && estimated > total
})

// Stacked progress computed values for RAM (current + estimated additional)
const totalRamBytes = computed(() => ramEstimate.value?.system_ram_total || systemStore.systemStatus.system?.memory?.total || 0)
const currentRamBytes = computed(() => ramEstimate.value?.system_ram_used || systemStore.systemStatus.system?.memory?.used || 0)
const estimatedRamBytes = computed(() => ramEstimate.value?.estimated_ram || 0)
const totalEstimatedRamBytes = computed(() => {
  // Total estimated RAM = current system usage + estimated additional RAM
  return currentRamBytes.value + estimatedRamBytes.value
})
const currentRamPercent = computed(() => {
  const total = totalRamBytes.value || 1
  return Math.min(100, Math.max(0, Math.round((currentRamBytes.value / total) * 100)))
})
const additionalRamPercent = computed(() => {
  const total = totalRamBytes.value || 1
  const add = ramEstimate.value?.estimated_ram || 0
  // Cap so current + additional does not exceed 100
  const pct = Math.round((add / total) * 100)
  return Math.max(0, Math.min(100 - currentRamPercent.value, pct))
})

// Stacked progress computed values for VRAM (current + estimated additional)
const totalVramBytes = computed(() => {
  if (realtimeVramData.value) {
    if (typeof realtimeVramData.value.total === 'number') {
      return realtimeVramData.value.total
    }
    if (typeof realtimeVramData.value.total_vram === 'number') {
      return realtimeVramData.value.total_vram
    }
  }
  return systemStore.gpuInfo.total_vram || 0
})
const currentVramBytes = computed(() => {
  if (realtimeVramData.value && typeof realtimeVramData.value.used_vram === 'number') {
    return realtimeVramData.value.used_vram
  }
  const total = systemStore.gpuInfo.total_vram || 0
  const available = systemStore.gpuInfo.available_vram
  if (total && typeof available === 'number') {
    return Math.max(0, total - available)
  }
  return 0
})
const estimatedVramBytes = computed(() => vramEstimate.value?.estimated_vram || 0)
const totalEstimatedVramBytes = computed(() => {
  // Total estimated VRAM = current GPU usage + estimated additional VRAM
  return currentVramBytes.value + estimatedVramBytes.value
})
const currentVramPercent = computed(() => {
  const total = totalVramBytes.value || 1
  return Math.min(100, Math.max(0, Math.round(((currentVramBytes.value || 0) / total) * 100)))
})
const additionalVramPercent = computed(() => {
  const total = totalVramBytes.value || 1
  const add = vramEstimate.value?.estimated_vram || 0
  const pct = Math.round((add / total) * 100)
  return Math.max(0, Math.min(100 - currentVramPercent.value, pct))
})

// RAM Status indicators for new dashboard
const ramStatus = computed(() => {
  if (!ramEstimate.value && !realtimeRamData.value) return 'unknown'
  
  let usagePercent = 0
  if (ramEstimate.value?.system_ram_total) {
    usagePercent = ((totalEstimatedRamBytes.value / ramEstimate.value.system_ram_total) * 100)
  } else if (realtimeRamData.value?.percent) {
    usagePercent = realtimeRamData.value.percent
  }
  
  if (usagePercent < 70) return 'good'
  if (usagePercent < 90) return 'warning'
  return 'critical'
})

const ramStatusClass = computed(() => {
  const status = ramStatus.value
  if (status === 'good') return 'status-good'
  if (status === 'warning') return 'status-warning'
  if (status === 'critical') return 'status-critical'
  return 'status-unknown'
})

const ramStatusText = computed(() => {
  const status = ramStatus.value
  if (status === 'good') return 'Fits Comfortably'
  if (status === 'warning') return 'Tight Fit'
  if (status === 'critical') return 'Won\'t Fit'
  return 'Unknown'
})

const ramProgressText = computed(() => {
  const current = currentRamPercent.value
  const additional = additionalRamPercent.value
  return `${current}% used + ${additional}% est • ${formatFileSize(totalEstimatedRamBytes.value)} total`
})

const ramStatusMessage = computed(() => {
  const status = ramStatus.value
  if (status === 'good') {
    const available = totalRamBytes.value - totalEstimatedRamBytes.value
    return `✅ Fits Comfortably - ${formatFileSize(available)} buffer remaining`
  }
  if (status === 'warning') {
    return '⚠️ Memory usage is high - consider reducing context size or batch size'
  }
  if (status === 'critical') {
    return '❌ Usage exceeds available memory - configuration will not work'
  }
  return 'Loading memory information...'
})

// VRAM Status indicators for new dashboard
const vramStatus = computed(() => {
  if (!vramEstimate.value && !realtimeVramData.value) return 'unknown'
  
  const usagePercent = totalVramBytes.value > 0 
    ? (totalEstimatedVramBytes.value / totalVramBytes.value) * 100
    : 0
  
  if (usagePercent < 70) return 'good'
  if (usagePercent < 90) return 'warning'
  return 'critical'
})

const vramStatusClass = computed(() => {
  const status = vramStatus.value
  if (status === 'good') return 'status-good'
  if (status === 'warning') return 'status-warning'
  if (status === 'critical') return 'status-critical'
  return 'status-unknown'
})

const vramStatusText = computed(() => {
  const status = vramStatus.value
  if (status === 'good') return 'Fits Comfortably'
  if (status === 'warning') return 'Tight Fit'
  if (status === 'critical') return 'Won\'t Fit'
  return 'Unknown'
})

const vramProgressText = computed(() => {
  const current = currentVramPercent.value
  const additional = additionalVramPercent.value
  return `${current}% used + ${additional}% est • ${formatFileSize(totalEstimatedVramBytes.value)} total`
})

const vramStatusMessage = computed(() => {
  const status = vramStatus.value
  if (status === 'good') {
    const available = totalVramBytes.value - totalEstimatedVramBytes.value
    return `✅ Fits Comfortably - ${formatFileSize(available)} VRAM available`
  }
  if (status === 'warning') {
    return '⚠️ VRAM usage is high - consider reducing GPU layers or batch size'
  }
  if (status === 'critical') {
    return '❌ Usage exceeds available VRAM - reduce GPU layers or context size'
  }
  return 'Loading VRAM information...'
})

// Maximum values from backend recommendations
const maxGpuLayers = computed(() => {
  return modelRecommendations.value?.gpu_layers?.max || modelLayerInfo.value?.layer_count || 32
})

const maxContextSize = computed(() => {
  return modelRecommendations.value?.context_size?.max || modelLayerInfo.value?.context_length || 131072
})

const maxBatchSize = computed(() => {
  return modelRecommendations.value?.batch_size?.max || 512
})

// Maximum parallel processing based on attention head count
const maxParallel = computed(() => {
  return modelRecommendations.value?.parallel?.max || 8
})

const maxTopK = computed(() => {
  return modelRecommendations.value?.top_k?.max || 200
})

// Recommended values from backend
const recommendedGpuLayers = computed(() => modelRecommendations.value?.gpu_layers?.recommended_value)
const recommendedContextSize = computed(() => modelRecommendations.value?.context_size?.recommended_value)
const recommendedBatchSize = computed(() => modelRecommendations.value?.batch_size?.recommended_value)
const recommendedTemperature = computed(() => modelRecommendations.value?.temperature?.recommended_value)
const recommendedTopK = computed(() => modelRecommendations.value?.top_k?.recommended_value)
const recommendedTopP = computed(() => modelRecommendations.value?.top_p?.recommended_value)

// Load supported flags from the active candle runtime (placeholder)
const loadSupportedFlags = async () => {
  try {
    const response = await fetch('/api/models/supported-flags')
    if (response.ok) {
      const data = await response.json()
      supportedFlags.value = data.supported_flags || []
      supportedConfigKeys.value = data.supported_config_keys || {}
    } else {
      console.warn('Failed to load supported flags:', response.statusText)
      supportedFlags.value = []
      supportedConfigKeys.value = {}
    }
  } catch (error) {
    console.error('Error loading supported flags:', error)
    supportedFlags.value = []
    supportedConfigKeys.value = {}
  }
}

// Computed properties to check if specific flags are supported
const isTypicalPSupported = computed(() => supportedConfigKeys.value.typical_p === true)
const isMinPSupported = computed(() => supportedConfigKeys.value.min_p === true)
const isTfsZSupported = computed(() => supportedConfigKeys.value.tfs_z === true)
const isPresencePenaltySupported = computed(() => supportedConfigKeys.value.presence_penalty === true)
const isFrequencyPenaltySupported = computed(() => supportedConfigKeys.value.frequency_penalty === true)
const isJsonSchemaSupported = computed(() => supportedConfigKeys.value.json_schema === true)
const isCacheTypeVSupported = computed(() => supportedConfigKeys.value.cache_type_v === true)

// Individual field validation
const gpuLayersValidation = computed(() => {
  if (!modelLayerInfo.value) return null
  if (config.value.n_gpu_layers > modelLayerInfo.value.layer_count) {
    return {
      type: 'error',
      message: `Exceeds model's ${modelLayerInfo.value.layer_count} layers`
    }
  }
  if (config.value.n_gpu_layers === modelLayerInfo.value.layer_count && modelLayerInfo.value.layer_count > 0) {
    return {
      type: 'success',
      message: 'Fully offloaded to GPU'
    }
  }
  return null
})

const contextSizeValidation = computed(() => {
  if (!modelLayerInfo.value) return null
  if (config.value.ctx_size > modelLayerInfo.value.context_length) {
    return {
      type: 'warning',
      message: `Exceeds model's max of ${modelLayerInfo.value.context_length.toLocaleString()}`
    }
  }
  return null
})

const batchSizeValidation = computed(() => {
  if (!modelLayerInfo.value) return null
  if (config.value.batch_size > maxBatchSize.value) {
    return {
      type: 'warning',
      message: `Exceeds recommended max of ${maxBatchSize.value}`
    }
  }
  return null
})

const parallelValidation = computed(() => {
  if (!modelLayerInfo.value) return null
  if (config.value.parallel > maxParallel.value) {
    return {
      type: 'warning',
      message: `Exceeds recommended max of ${maxParallel.value}`
    }
  }
  return null
})

// Configuration validation warnings
const configWarnings = computed(() => {
  const warnings = []

  if (modelLayerInfo.value) {
    // Check context size
    if (config.value.ctx_size > modelLayerInfo.value.context_length) {
      warnings.push({
        type: 'warning',
        message: `Context size (${config.value.ctx_size}) exceeds model's maximum context length (${modelLayerInfo.value.context_length})`,
        field: 'ctx_size'
      })
    }

    // Check batch size
    if (config.value.batch_size > maxBatchSize.value) {
      warnings.push({
        type: 'warning',
        message: `Batch size (${config.value.batch_size}) exceeds recommended maximum (${maxBatchSize.value}) based on model architecture`,
        field: 'batch_size'
      })
    }

    // Check GPU layers
    if (config.value.n_gpu_layers > modelLayerInfo.value.layer_count) {
      warnings.push({
        type: 'error',
        message: `GPU layers (${config.value.n_gpu_layers}) exceeds model's total layers (${modelLayerInfo.value.layer_count})`,
        field: 'n_gpu_layers'
      })
    }

    // Check parallel processing
    if (config.value.parallel > maxParallel.value) {
      warnings.push({
        type: 'warning',
        message: `Parallel processing (${config.value.parallel}) exceeds recommended maximum (${maxParallel.value}) based on attention head count`,
        field: 'parallel'
      })
    }
  }

  return warnings
})

onMounted(async () => {
  await buildsStore.fetchBuilds()
  await loadModel()
  await systemStore.fetchSystemStatus()
  await loadSupportedFlags()
  initializeConfig()

  // Subscribe to unified monitoring updates for real-time memory data
  wsStore.subscribeToUnifiedMonitoring((data) => {
    // Extract RAM data from unified stream
    if (data.system?.memory) {
      realtimeRamData.value = {
        total: data.system.memory.total,
        available: data.system.memory.available,
        used: data.system.memory.used,
        percent: data.system.memory.percent,
        free: data.system.memory.free,
        cached: data.system.memory.cached,
        buffers: data.system.memory.buffers,
        swap_total: data.system.memory.swap_total,
        swap_used: data.system.memory.swap_used,
        timestamp: Date.now()
      }
    }

    // Extract VRAM data from unified stream
    if (data.gpu?.vram_data) {
      realtimeVramData.value = normalizeVramData(data.gpu.vram_data)
    }

    // Extract running instance data for performance metrics
  })

  // Automatically trigger initial estimates if no real-time data yet
  if (!realtimeRamData.value) {
    estimateRam()
  }
  if (!realtimeVramData.value && !systemStore.gpuInfo.cpu_only_mode) {
    estimateVram()
  }
})

onUnmounted(() => {
  // WebSocket subscriptions are automatically cleaned up by the store
  if (vramEstimateTimeout) {
    clearTimeout(vramEstimateTimeout)
    vramEstimateTimeout = null
  }
  if (ramEstimateTimeout) {
    clearTimeout(ramEstimateTimeout)
    ramEstimateTimeout = null
  }
  if (vramAbortController) {
    vramAbortController.abort()
    vramAbortController = null
  }
  if (ramAbortController) {
    ramAbortController.abort()
    ramAbortController = null
  }
})

const loadModel = async () => {
  const modelId = route.params.id
  if (modelId) {
    try {
      await modelStore.fetchModels()
      // Find the specific quantization in the grouped structure
      model.value = modelStore.allQuantizations.find(m => m.id === parseInt(modelId))
      if (!model.value) {
        toast.error('The requested model quantization could not be found')
        router.push('/models')
        return
      }

      // Load model layer information
      await loadModelLayerInfo()
      
      // Load model recommendations
      await loadModelRecommendations()

    } catch (error) {
      console.error('Failed to load model:', error)
      toast.error('Failed to load model')
    }
  }
}

const loadModelLayerInfo = async () => {
  if (!model.value) return

  layerInfoLoading.value = true
  try {
    const response = await fetch(`/api/models/${model.value.id}/layer-info`)
    if (response.ok) {
      modelLayerInfo.value = await response.json()
      console.log('Model layer info:', modelLayerInfo.value)
    } else {
      console.warn('Failed to load model layer info, using defaults')
      modelLayerInfo.value = { layer_count: 32 }
    }
  } catch (error) {
    console.error('Error loading model layer info:', error)
    modelLayerInfo.value = { layer_count: 32 }
  } finally {
    layerInfoLoading.value = false
  }
}

const loadModelRecommendations = async () => {
  if (!model.value) return

  recommendationsLoading.value = true
  try {
    const response = await fetch(`/api/models/${model.value.id}/recommendations`)
    if (response.ok) {
      modelRecommendations.value = await response.json()
      console.log('Model recommendations:', modelRecommendations.value)
    } else {
      console.warn('Failed to load model recommendations')
      modelRecommendations.value = null
    }
  } catch (error) {
    console.error('Error loading model recommendations:', error)
    modelRecommendations.value = null
  } finally {
    recommendationsLoading.value = false
  }
}

const regenerateModelInfo = async () => {
  if (!model.value) return

  regeneratingInfo.value = true
  try {
    const response = await fetch(`/api/models/${model.value.id}/regenerate-info`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    if (response.ok) {
      const result = await response.json()
      toast.success('Model information regenerated successfully')

      // Update model from store if available
      if (modelStore) {
        await modelStore.fetchModels()
        // Reload the current model
        model.value = modelStore.allQuantizations.find(m => m.id === model.value.id)
      }

      // Reload layer info to get updated metadata
      await loadModelLayerInfo()
      
      // Reload recommendations with updated metadata
      await loadModelRecommendations()

      console.log('Regenerated model info:', result)
    } else {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      toast.error(`Failed to regenerate model info: ${error.detail || 'Unknown error'}`)
    }
  } catch (error) {
    console.error('Error regenerating model info:', error)
    toast.error('Failed to regenerate model information')
  } finally {
    regeneratingInfo.value = false
  }
}

const parseModelConfig = (rawConfig) => {
  if (!rawConfig) return null
  if (typeof rawConfig === 'string') {
    try {
      return JSON.parse(rawConfig)
    } catch (error) {
      console.error('Failed to parse model config string:', error)
      return null
    }
  }
  if (typeof rawConfig === 'object') {
    return rawConfig
  }
  return null
}

const initializeConfig = () => {
  const defaults = getDefaultConfig()
  const loadedConfig = parseModelConfig(model.value?.config)
  if (loadedConfig) {
    config.value = { ...defaults, ...loadedConfig }
    topologyPlan.value = loadedConfig.topology_plan || null
    for (const key in defaults) {
      if (config.value[key] === undefined || config.value[key] === null) {
        config.value[key] = defaults[key]
      }
      if (typeof defaults[key] === 'boolean' && typeof config.value[key] !== 'boolean') {
        config.value[key] = Boolean(config.value[key])
      }
      if (typeof defaults[key] === 'number' && typeof config.value[key] !== 'number') {
        const num = Number(config.value[key])
        config.value[key] = Number.isNaN(num) ? defaults[key] : num
      }
    }
    normalizeRuntimeConfig()
  } else {
    config.value = { ...defaults }
    topologyPlan.value = null
    normalizeRuntimeConfig()
  }
}

const getDefaultConfig = () => ({
  host: '0.0.0.0',
  port: DEFAULT_CANDLE_PORT,
  timeout: 300,
  build_name: null,
  weights_path: '',
  weights_file: '',
  kvcache_mem_gpu: 4096,
  kvcache_mem_cpu: null,
  max_num_seqs: null,
  block_size: null,
  hf_token: '',
  hf_token_path: '',
  verbose: false,
  cpu: false,
  record_conversation: false,
  holding_time: null,
  multithread: false,
  log: false,
  temperature: null,
  top_p: null,
  min_p: null,
  top_k: null,
  frequency_penalty: null,
  presence_penalty: null,
  prefill_chunk_size: null,
  fp8_kvcache: false,
  dtype: null,
  isq: null,
  max_gen_tokens: 4096,
  device_id: 0,
  features: [],
  extra_args: [],
  env: {},
  build_profile: 'release'
})

// KV cache options
const kvCacheOptions = [
  { label: 'FP32 (full precision)', value: 'f32' },
  { label: 'FP16 (half precision)', value: 'f16' },
  { label: 'BF16 (bfloat16)', value: 'bf16' },
  { label: 'Q8_0 (8-bit)', value: 'q8_0' },
  { label: 'Q5_1 (5-bit high quality)', value: 'q5_1' },
  { label: 'Q5_0 (5-bit)', value: 'q5_0' },
  { label: 'Q4_1 (4-bit high quality)', value: 'q4_1' },
  { label: 'Q4_0 (4-bit)', value: 'q4_0' },
  { label: 'IQ4_NL (4-bit non-linear)', value: 'iq4_nl' }
]

// MoE offload patterns
const moeOffloadPatterns = [
  { label: 'None', value: 'none' },
  { label: 'All MoE layers to CPU', value: 'all' },
  { label: 'Up/Down projections to CPU', value: 'up_down' },
  { label: 'Up projection only to CPU', value: 'up' },
  { label: 'Custom pattern', value: 'custom' }
]

// Apply architecture preset
const selectedPreset = ref(null)
const applyPreset = async (presetName, skipPreview = false) => {
  if (!model.value) return

  try {
    const response = await fetch(`/api/models/${model.value.id}/architecture-presets`)
    if (!response.ok) throw new Error('Failed to fetch presets')

    const data = await response.json()
    const preset = data.presets[presetName]

    if (preset) {
      // Calculate changes
      const changes = calculateChanges(preset, config.value)
      
      if (!skipPreview && changes.length > 0) {
        // Show preview
        previewData.value = {
          type: 'preset',
          presetName: presetName,
          changes: changes,
          impact: null,
          newConfig: { ...config.value, ...preset }
        }
        showPreview.value = true
        return
      }

      // Apply directly if skipPreview is true
      selectedPreset.value = presetName
      Object.assign(config.value, preset)
      normalizeRuntimeConfig()

      // Re-estimate memory to reflect preset changes
      await estimateVram()
      await estimateRam()

      toast.success(`${presetName.charAt(0).toUpperCase() + presetName.slice(1)} preset applied`)
    }
  } catch (error) {
    toast.error('Failed to apply preset')
  }
}

// Handle MoE pattern change
const handleMoEPatternChange = () => {
  // Automatically set the custom pattern based on selection
  switch (config.value.moe_offload_pattern) {
    case 'all':
      config.value.moe_offload_custom = '.ffn_.*_exps.=CPU'
      break
    case 'up_down':
      config.value.moe_offload_custom = '.ffn_(up|down)_exps.=CPU'
      break
    case 'up':
      config.value.moe_offload_custom = '.ffn_(up)_exps.=CPU'
      break
    case 'none':
      config.value.moe_offload_custom = ''
      break
    case 'custom':
      // User will input custom pattern
      break
  }
}

// Watch for flash attention changes to warn about V cache
watch(() => config.value.flash_attn, (newVal) => {
  // Ensure cache_type_v has a safe value based on flash_attn state
  if (!newVal) {
    // When Flash Attention is disabled, V cache should be null or f16
    if (!config.value.cache_type_v || typeof config.value.cache_type_v !== 'string') {
      config.value.cache_type_v = null
    }
  } else {
    // When Flash Attention is enabled, ensure V cache has a valid string value
    if (!config.value.cache_type_v || typeof config.value.cache_type_v !== 'string') {
      config.value.cache_type_v = config.value.cache_type_k || 'f16'
    }
  }
})

// Handle wizard-generated configuration
const handleWizardConfig = async (wizardConfig) => {
  try {
    if (!model.value) {
      toast.error('No model selected')
      return
    }

    // Apply wizard configuration
    const defaults = getDefaultConfig()
    Object.assign(config.value, defaults, wizardConfig)
    normalizeRuntimeConfig()

    // Re-estimate memory to reflect config changes
    await estimateVram()
    await estimateRam()

    // Automatically save to backend
    saveLoading.value = true
    try {
      await modelStore.updateModelConfig(model.value.id, config.value)
      
      // Reload model to get updated config
      await loadModel()
      
      toast.success('Configuration wizard settings applied and saved successfully')
      showWizard.value = false
    } catch (saveError) {
      console.error('Save config error:', saveError)
      toast.error('Configuration applied but failed to save. Please click Save Config manually.')
    } finally {
      saveLoading.value = false
    }
  } catch (error) {
    console.error('Error applying wizard config:', error)
    toast.error('Failed to apply wizard configuration')
  }
}

const generateAutoConfig = async (skipPreview = false) => {
  autoConfigLoading.value = true
  smartAutoDebug.value = null
  try {
    if (!model.value) {
      toast.error('No model selected')
      return
    }

    const payload = {
      usage_mode: smartAutoUsageMode.value,
      speed_quality: smartAutoSpeedQuality.value,
      use_case: smartAutoUseCase.value,
      debug: smartAutoIncludeDebug.value,
      restrict_to_nvlink: smartAutoRestrictNvlink.value
    }

    if (selectedPreset.value) {
      payload.preset = selectedPreset.value
    }

    const target = toOptionalInt(smartAutoTargetConcurrency.value)
    if (target !== null) {
      const clampedTarget = Math.max(1, Math.min(16, target))
      payload.target_concurrency = clampedTarget
      smartAutoTargetConcurrency.value = clampedTarget
    }

    const tokensHint = toOptionalInt(smartAutoMaxTokensHint.value)
    if (tokensHint !== null) {
      const clampedTokens = Math.max(256, tokensHint)
      payload.max_tokens_hint = clampedTokens
      smartAutoMaxTokensHint.value = clampedTokens
    }

    const response = await fetch(`/api/models/${model.value.id}/smart-auto`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })

    if (!response.ok) {
      throw new Error(`Smart auto failed: ${response.statusText}`)
    }

    const responseJson = await response.json()
    const smartConfig = responseJson?.config ?? responseJson
    const plan = responseJson?.plan ?? smartConfig?.topology_plan ?? null
    const debugInfo = responseJson?.debug ?? null
    smartAutoDebug.value = debugInfo
    topologyPlan.value = plan

    const defaults = getDefaultConfig()
    const newConfig = { ...defaults, ...smartConfig }

    const changes = calculateChanges(newConfig, config.value)
    const impact = await calculateImpact(newConfig, config.value)

    const shouldPreview = !skipPreview && (changes.length > 0 || !!debugInfo || !!plan)
    if (shouldPreview) {
      previewData.value = {
        type: 'smart-auto',
        presetName: selectedPreset.value || '',
        changes,
        impact,
        newConfig,
        plan,
        debug: debugInfo
      }
      showPreview.value = true
      return
    }

    config.value = newConfig
    topologyPlan.value = plan

    for (const key in defaults) {
      if (config.value[key] === undefined || config.value[key] === null) {
        config.value[key] = defaults[key]
      }
      if (typeof defaults[key] === 'boolean' && typeof config.value[key] !== 'boolean') {
        config.value[key] = Boolean(config.value[key])
      }
      if (typeof defaults[key] === 'number' && typeof config.value[key] !== 'number') {
        const num = Number(config.value[key])
        config.value[key] = isNaN(num) ? defaults[key] : num
      }
    }
    normalizeRuntimeConfig()

    const isCpuOnlyMode = systemStore.gpuInfo.cpu_only_mode
    const optimizationType = isCpuOnlyMode ? 'CPU-optimized' : 'GPU-optimized'

    toast.success(`${optimizationType} configuration generated successfully`)

    await estimateVram()
    await estimateRam()
  } catch (error) {
    console.error('Smart auto error:', error)
    toast.error('Failed to generate automatic configuration')
  } finally {
    autoConfigLoading.value = false
  }
}

const estimateVram = async () => {
  if (!model.value) return

  if (vramAbortController) {
    vramAbortController.abort()
  }

  const controller = new AbortController()
  vramAbortController = controller
  vramLoading.value = true

  try {
    const response = await fetch('/api/models/vram-estimate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_id: model.value.id,
        config: config.value,
        usage_mode: smartAutoUsageMode.value
      }),
      signal: controller.signal
    })

    if (!response.ok) {
      throw new Error('VRAM estimation failed')
    }

    const data = await response.json()
    vramEstimate.value = data
  } catch (error) {
    if (error?.name === 'AbortError') {
      return
    }
    console.error('VRAM estimation error:', error)
    toast.error('Could not estimate VRAM usage')
  } finally {
    if (vramAbortController === controller) {
      vramLoading.value = false
      vramAbortController = null
    }
  }
}

const estimateRam = async () => {
  if (!model.value) return

  if (ramAbortController) {
    ramAbortController.abort()
  }

  const controller = new AbortController()
  ramAbortController = controller
  ramLoading.value = true

  try {
    const response = await fetch('/api/models/ram-estimate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_id: model.value.id,
        config: config.value,
        usage_mode: smartAutoUsageMode.value
      }),
      signal: controller.signal
    })

    if (!response.ok) {
      throw new Error('RAM estimation failed')
    }

    ramEstimate.value = await response.json()
  } catch (error) {
    if (error?.name === 'AbortError') {
      return
    }
    console.error('RAM estimation error:', error)
    toast.error('Could not estimate RAM usage')
  } finally {
    if (ramAbortController === controller) {
      ramLoading.value = false
      ramAbortController = null
    }
  }
}

const updateVramEstimate = (force = false) => {
  if (force || !vramEstimate.value) {
    return estimateVram()
  }
  if (vramEstimateTimeout) {
    clearTimeout(vramEstimateTimeout)
  }
  vramEstimateTimeout = setTimeout(() => {
    estimateVram()
  }, ESTIMATE_DEBOUNCE_MS)
  return Promise.resolve()
}

const updateRamEstimate = (force = false) => {
  if (force || !ramEstimate.value) {
    return estimateRam()
  }
  if (ramEstimateTimeout) {
    clearTimeout(ramEstimateTimeout)
  }
  ramEstimateTimeout = setTimeout(() => {
    estimateRam()
  }, ESTIMATE_DEBOUNCE_MS)
  return Promise.resolve()
}

const saveConfig = async () => {
  if (!model.value) return

  normalizeRuntimeConfig()
  saveLoading.value = true
  try {
    await modelStore.updateModelConfig(model.value.id, config.value)

    toast.success('Model configuration has been updated')
  } catch (error) {
    console.error('Save config error:', error)
    toast.error('Failed to save configuration')
  } finally {
    saveLoading.value = false
  }
}

const formatFileSize = (bytes) => {
  if (!bytes) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// Calculate changes between old and new config
const calculateChanges = (newConfig, oldConfig) => {
  const changes = []
  const importantFields = {
    build_name: 'Build',
    weights_path: 'Weights Directory',
    weights_file: 'Weights File',
    host: 'Host',
    port: 'Port',
    kvcache_mem_gpu: 'KV Cache GPU Memory',
    kvcache_mem_cpu: 'KV Cache CPU Memory',
    dtype: 'Runtime DType',
    isq: 'In-situ Quantization',
    max_gen_tokens: 'Max Generated Tokens',
    max_num_seqs: 'Max Concurrent Sequences',
    block_size: 'KV Block Size',
    hf_token_path: 'HF Token Path',
    verbose: 'Verbose Logging',
    cpu: 'CPU Mode',
    record_conversation: 'Record Conversation',
    holding_time: 'Holding Time',
    multithread: 'Multithread Scheduler',
    log: 'Structured Logging',
    temperature: 'Temperature',
    top_p: 'Top-P',
    min_p: 'Min-P',
    top_k: 'Top-K',
    frequency_penalty: 'Frequency Penalty',
    presence_penalty: 'Presence Penalty',
    prefill_chunk_size: 'Prefill Chunk Size',
    fp8_kvcache: 'FP8 KV Cache',
    n_gpu_layers: 'GPU Layers',
    ctx_size: 'Context Size',
    batch_size: 'Batch Size',
    threads: 'CPU Threads',
    flash_attn: 'Flash Attention',
    cont_batching: 'Continuous Batching'
  }

  for (const [key, label] of Object.entries(importantFields)) {
    const oldValue = oldConfig[key]
    const newValue = newConfig[key]
    
    if (oldValue !== newValue && (oldValue !== undefined || newValue !== undefined)) {
      changes.push({
        field: label,
        before: oldValue,
        after: newValue,
        description: getFieldDescription(key, oldValue, newValue)
      })
    }
  }

  return changes
}

// Get description for field change
const getFieldDescription = (key, oldValue, newValue) => {
  const descriptions = {
    build_name: 'Selected candle-vllm build to launch',
    weights_path: 'Updated weights directory used by the runtime',
    weights_file: 'Updated GGUF model file used for inference',
    host: 'Adjusted host binding for the runtime server',
    port: 'Updated runtime API port',
    kvcache_mem_gpu: 'Adjusted GPU memory reserved for KV cache',
    kvcache_mem_cpu: 'Adjusted CPU memory reserved for KV cache spillover',
    dtype: 'Changed runtime tensor data type',
    isq: 'Updated in-situ quantization mode',
    max_gen_tokens: 'Adjusted maximum generated token count',
    max_num_seqs: 'Changed the number of concurrent sequences candle can schedule',
    block_size: 'Adjusted KV cache block size for scheduling',
    hf_token_path: 'Updated Hugging Face token path',
    verbose: newValue ? 'Enabled verbose request logging' : 'Disabled verbose request logging',
    cpu: newValue ? 'Enabled CPU-only execution mode' : 'Disabled CPU-only execution mode',
    record_conversation: newValue ? 'Enabled conversation recording' : 'Disabled conversation recording',
    holding_time: 'Adjusted request batching holding time',
    multithread: newValue ? 'Enabled multithreaded scheduler' : 'Disabled multithreaded scheduler',
    log: newValue ? 'Enabled structured logging output' : 'Disabled structured logging output',
    temperature: oldValue < newValue 
      ? 'Higher temperature for more creative outputs' 
      : 'Lower temperature for more focused outputs',
    top_p: 'Updated nucleus sampling probability mass',
    min_p: 'Updated minimum probability mass',
    top_k: 'Updated top-k sampling cutoff',
    frequency_penalty: 'Adjusted frequency penalty for repeated tokens',
    presence_penalty: 'Adjusted presence penalty for repeated tokens',
    prefill_chunk_size: 'Adjusted prefill chunk size for batching',
    fp8_kvcache: newValue ? 'Enabled FP8 KV cache format' : 'Disabled FP8 KV cache format',
    n_gpu_layers: oldValue < newValue 
      ? 'More layers offloaded to GPU for faster inference' 
      : 'Fewer layers offloaded to GPU to reduce VRAM usage',
    ctx_size: oldValue < newValue 
      ? 'Increased context window for longer conversations' 
      : 'Reduced context window to save memory',
    batch_size: oldValue < newValue 
      ? 'Increased batch size for better throughput' 
      : 'Reduced batch size to save memory'
  }
  return descriptions[key] || ''
}

// Calculate impact of config changes
const calculateImpact = async (newConfig, oldConfig) => {
  const impact = {}
  
  // Estimate VRAM changes
  const oldVram = vramEstimate.value?.estimated_vram || 0
  const newVram = await estimateVramForConfig(newConfig)
  
  if (oldVram && newVram) {
    const diff = newVram - oldVram
    if (Math.abs(diff) > 1000000000) { // > 1GB
      const diffPercent = ((diff / oldVram) * 100).toFixed(1)
      impact.vram = diff > 0 
        ? `VRAM Usage: ${formatFileSize(oldVram)} → ${formatFileSize(newVram)} (+${diffPercent}%)`
        : `VRAM Usage: ${formatFileSize(oldVram)} → ${formatFileSize(newVram)} (${diffPercent}%)`
    }
  }

  // Estimate performance impact
  const oldLayers = oldConfig.n_gpu_layers || 0
  const newLayers = newConfig.n_gpu_layers || 0
  if (newLayers > oldLayers) {
    const layerIncrease = newLayers - oldLayers
    const estimatedSpeedup = Math.min(50, Math.round((layerIncrease / 32) * 50))
    if (estimatedSpeedup > 5) {
      impact.performance = `Performance: ~${estimatedSpeedup}% faster with more GPU layers`
    }
  }

  return Object.keys(impact).length > 0 ? impact : null
}

// Estimate VRAM for a config without applying it
const estimateVramForConfig = async (testConfig) => {
  if (!model.value) return 0
  
  try {
    const response = await fetch('/api/models/vram-estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: model.value.id,
        config: testConfig
      })
    })
    if (response.ok) {
      const estimate = await response.json()
      return estimate.estimated_vram || 0
    }
  } catch (error) {
    console.error('VRAM estimation error:', error)
  }
  return vramEstimate.value?.estimated_vram || 0
}

// Apply preview changes
const applyPreviewChanges = async () => {
  if (!previewData.value) return
  
  previewApplying.value = true
  try {
    // Apply the new config
    config.value = previewData.value.newConfig
    topologyPlan.value = previewData.value.plan ?? topologyPlan.value
    
    // Ensure all fields have safe values
    const defaults = getDefaultConfig()
    for (const key in defaults) {
      if (config.value[key] === undefined || config.value[key] === null) {
        config.value[key] = defaults[key]
      }
      // Type coercion for critical fields
      if (typeof defaults[key] === 'boolean' && typeof config.value[key] !== 'boolean') {
        config.value[key] = Boolean(config.value[key])
      }
      if (typeof defaults[key] === 'number' && typeof config.value[key] !== 'number') {
        const num = Number(config.value[key])
        config.value[key] = isNaN(num) ? defaults[key] : num
      }
    }
    normalizeRuntimeConfig()
    
    // Re-estimate memory
    await estimateVram()
    await estimateRam()
    
    // Save the preset name and close preview
    const previewType = previewData.value.type
    const presetName = previewData.value.presetName
    showPreview.value = false
    previewData.value = null
    
    // Show success message
    if (previewType === 'preset' && presetName) {
      selectedPreset.value = presetName
      toast.success(`${presetName.charAt(0).toUpperCase() + presetName.slice(1)} preset applied`)
    } else {
      const isCpuOnlyMode = systemStore.gpuInfo.cpu_only_mode
      const optimizationType = isCpuOnlyMode ? 'CPU-optimized' : 'GPU-optimized'
      toast.success(`${optimizationType} configuration applied successfully`)
    }
  } catch (error) {
    console.error('Error applying changes:', error)
    toast.error('Failed to apply configuration changes')
  } finally {
    previewApplying.value = false
  }
}

// Tooltip generators with architecture-specific recommendations
const getTemperatureTooltip = () => {
  const architecture = modelLayerInfo.value?.architecture?.toLowerCase() || ''
  let baseMsg = 'Controls randomness (0.1=deterministic, 2.0=creative)'

  if (architecture.includes('glm')) {
    baseMsg += ' | GLM: Recommended 1.0'
  } else if (architecture.includes('deepseek')) {
    baseMsg += ' | DeepSeek: Recommended 1.0'
  } else if (architecture.includes('qwen')) {
    baseMsg += ' | Qwen: Recommended 0.7'
  } else if (architecture.includes('codellama') || model.value?.name.toLowerCase().includes('code')) {
    baseMsg += ' | Coding: Recommended 0.1-0.7'
  }

  return baseMsg
}

const getTopKTooltip = () => {
  return `Top-K sampling (limit to top K tokens) | Recommended: 40 for GLM/DeepSeek, 50 for others`
}

const getTopPTooltip = () => {
  const architecture = modelLayerInfo.value?.architecture?.toLowerCase() || ''
  let baseMsg = 'Top-P (nucleus) sampling'

  if (architecture.includes('glm') || architecture.includes('deepseek')) {
    baseMsg += ' | Recommended: 0.95'
  } else if (architecture.includes('qwen')) {
    baseMsg += ' | Recommended: 0.9-0.95'
  } else {
    baseMsg += ' | Recommended: 0.95'
  }

  return baseMsg
}

const getRepeatPenaltyTooltip = () => {
  let baseMsg = 'Penalty for repeating tokens (1.0=no penalty)'

  const ctxLength = modelLayerInfo.value?.context_length || 0
  if (ctxLength > 32768) {
    baseMsg += ' | Long context: Use 1.0-1.05'
  } else if (ctxLength < 2048) {
    baseMsg += ' | Short context: Use 1.1-1.2'
  } else {
    baseMsg += ' | Standard: Use 1.1'
  }

  return baseMsg
}

// Rich tooltip data for key settings
const gpuLayersTooltip = computed(() => {
  const rec = modelRecommendations.value?.gpu_layers
  if (!rec) {
    return {
      description: 'Controls how many model layers are offloaded to GPU. More layers = faster inference but higher VRAM usage.',
      whenToAdjust: 'Increase for faster generation if you have VRAM. Decrease if running out of VRAM.',
      tradeoffs: [
        'Higher values: Faster inference, better for long conversations',
        'Lower values: Lower VRAM usage, falls back to CPU which is slower'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Controls how many model layers are offloaded to GPU. More layers = faster inference but higher VRAM usage.',
    whenToAdjust: 'Increase for faster generation if you have VRAM. Decrease if running out of VRAM.',
    tradeoffs: [
      'Higher values: Faster inference, better for long conversations',
      'Lower values: Lower VRAM usage, falls back to CPU which is slower'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.value} layers: ${r.description}`)
  }
})

const contextSizeTooltip = computed(() => {
  const rec = modelRecommendations.value?.context_size
  if (!rec) {
    return {
      description: 'Maximum number of tokens the model can process in context. Higher = can remember more but uses more memory.',
      whenToAdjust: 'Increase for long conversations or documents. Decrease if running low on memory.',
      tradeoffs: [
        'Higher values: Can handle longer conversations, better for documents',
        'Lower values: Lower memory usage, faster processing'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Maximum number of tokens the model can process in context. Higher = can remember more but uses more memory.',
    whenToAdjust: 'Increase for long conversations or documents. Decrease if running low on memory.',
    tradeoffs: [
      'Higher values: Can handle longer conversations, better for documents',
      'Lower values: Lower memory usage, faster processing'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const temperatureTooltip = computed(() => {
  const rec = modelRecommendations.value?.temperature
  if (!rec) {
    return {
      description: 'Controls randomness in model responses. Lower = more focused and deterministic, Higher = more creative and varied.',
      whenToAdjust: 'Lower for code/technical tasks. Higher for creative writing. Adjust if outputs are too repetitive or too random.',
      tradeoffs: [
        'Low (0.1-0.3): Focused, deterministic, good for code',
        'Medium (0.7-1.0): Balanced, natural conversations',
        'High (1.5-2.0): Creative, varied, unpredictable'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Controls randomness in model responses. Lower = more focused and deterministic, Higher = more creative and varied.',
    whenToAdjust: 'Lower for code/technical tasks. Higher for creative writing. Adjust if outputs are too repetitive or too random.',
    tradeoffs: [
      'Low (0.1-0.3): Focused, deterministic, good for code',
      'Medium (0.7-1.0): Balanced, natural conversations',
      'High (1.5-2.0): Creative, varied, unpredictable'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const batchSizeTooltip = computed(() => {
  const rec = modelRecommendations.value?.batch_size
  if (!rec) {
    return {
      description: 'Number of tokens processed in parallel. Higher = faster but uses more memory.',
      whenToAdjust: 'Increase if you have VRAM available. Decrease if getting out-of-memory errors.',
      tradeoffs: [
        'Higher values: Faster token generation, better throughput',
        'Lower values: Lower memory usage, more sequential processing'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Number of tokens processed in parallel. Higher = faster but uses more memory.',
    whenToAdjust: 'Increase if you have VRAM available. Decrease if getting out-of-memory errors.',
    tradeoffs: [
      'Higher values: Faster token generation, better throughput',
      'Lower values: Lower memory usage, more sequential processing'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const topKTooltip = computed(() => {
  const rec = modelRecommendations.value?.top_k
  if (!rec) {
    return {
      description: 'Limits sampling to the top K most likely tokens. Lower = more focused, Higher = more diverse.',
      whenToAdjust: 'Lower for focused outputs. Higher for more variety. Works together with Top-P.',
      tradeoffs: [
        'Lower values (10-30): More focused, deterministic outputs',
        'Medium values (40-50): Balanced diversity (recommended)',
        'Higher values (100+): More random, less coherent outputs'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Limits sampling to the top K most likely tokens. Lower = more focused, Higher = more diverse.',
    whenToAdjust: 'Lower for focused outputs. Higher for more variety. Works together with Top-P.',
    tradeoffs: [
      'Lower values (10-30): More focused, deterministic outputs',
      'Medium values (40-50): Balanced diversity (recommended)',
      'Higher values (100+): More random, less coherent outputs'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const topPTooltip = computed(() => {
  const rec = modelRecommendations.value?.top_p
  if (!rec) {
    return {
      description: 'Nucleus sampling: considers tokens with cumulative probability mass up to P. Works with Top-K.',
      whenToAdjust: 'Lower for more focused outputs. Higher for more diversity. Typically keep at 0.9-0.95.',
      tradeoffs: [
        'Lower values (0.7-0.8): More conservative, focused sampling',
        'Medium values (0.9-0.95): Balanced diversity (recommended)',
        'Higher values (0.98-1.0): Includes very low-probability tokens'
      ],
      recommended: 'For this model: Loading recommendations...',
      ranges: []
    }
  }
  
  return {
    description: 'Nucleus sampling: considers tokens with cumulative probability mass up to P. Works with Top-K.',
    whenToAdjust: 'Lower for more focused outputs. Higher for more diversity. Typically keep at 0.9-0.95.',
    tradeoffs: [
      'Lower values (0.7-0.8): More conservative, focused sampling',
      'Medium values (0.9-0.95): Balanced diversity (recommended)',
      'Higher values (0.98-1.0): Includes very low-probability tokens'
    ],
    recommended: rec.description,
    ranges: rec.ranges.map(r => `${r.min}-${r.max}: ${r.description}`)
  }
})

const repeatPenaltyTooltip = computed(() => {
  const ctxLength = modelLayerInfo.value?.context_length || 0
  let recommended = '1.1 for standard contexts'
  let ranges = [
    '1.0: No penalty (allows repetition)',
    '1.1: Standard penalty (recommended)',
    '1.2: Strong penalty (prevents most repetition)'
  ]

  if (ctxLength > 32768) {
    recommended = '1.0-1.05 for long contexts'
    ranges = [
      '1.0-1.05: Minimal penalty (long context models)',
      '1.1: Standard (good for most cases)',
      '1.2+: Strong penalty (short contexts)'
    ]
  } else if (ctxLength < 2048) {
    recommended = '1.1-1.2 for short contexts'
  }

  return {
    description: 'Penalty applied to tokens that have appeared in the context. Higher = less repetition.',
    whenToAdjust: 'Increase if model repeats too much. Decrease if model avoids valid repetition. Adjust based on context length.',
    tradeoffs: [
      'Lower values (1.0-1.05): Allows natural repetition, good for long contexts',
      'Medium values (1.1): Balanced, prevents excessive repetition (recommended)',
      'Higher values (1.2-2.0): Strong prevention, may avoid valid repetition'
    ],
    recommended: recommended,
    ranges: ranges
  }
})

// Search functionality - computed properties to check section/field matches
// Tab matching for search - find which tab contains matching content
const matchingTabIndex = computed(() => {
  const query = configSearchQuery.value.toLowerCase().trim()
  if (!query) return null
  
  const searchTerms = query.split(/\s+/)
  
  for (let i = 0; i < tabLabels.length; i++) {
    const tab = tabLabels[i]
    const matchesAny = searchTerms.some(term => 
      tab.keywords.some(keyword => keyword.includes(term) || term.includes(keyword))
    )
    if (matchesAny) return i
  }
  
  return null
})

// Auto-switch to matching tab when searching
watch(matchingTabIndex, (tabIndex) => {
  if (tabIndex !== null && tabIndex !== undefined) {
    activeTabIndex.value = tabIndex
  }
})

// Check if current model is running
// Handle stop model
const handleStopModel = async () => {
  if (!model.value) return
  
  try {
    await modelStore.stopModel(model.value.id)
    toast.success('Model stopped successfully')
  } catch (error) {
    console.error('Error stopping model:', error)
    toast.error('Failed to stop model')
  }
}

// Watch for config changes to update estimates
watch(config, () => {
  updateVramEstimate()
  updateRamEstimate()
}, { deep: true })

// Watch for model changes to update running state
// Handle onboarding tour
const handleOnboardingComplete = () => {
  localStorage.setItem('model-config-onboarding-completed', 'true')
  showOnboarding.value = false
}

const handleOnboardingSkip = () => {
  showOnboarding.value = false
  // Don't mark as completed if skipped
}

// Check if model has no configuration (empty state)
const hasNoConfig = computed(() => {
  if (!model.value) return false
  if (!model.value.config) return true
  const parsed = parseModelConfig(model.value.config)
  if (!parsed || typeof parsed !== 'object') return true
  const defaults = getDefaultConfig()
  const hasNonDefaults = Object.keys(parsed).some(key => {
    const value = parsed[key]
    const defaultValue = defaults[key]
    if (value === null || value === undefined || value === '') return false
    if (typeof defaultValue === 'number') {
      const numericValue = Number(value)
      return !Number.isNaN(numericValue) && numericValue !== defaultValue
    }
    return value !== defaultValue
  })
  return !hasNonDefaults
})

const hasLegacyFields = computed(() => {
  if (!model.value || !model.value.config) return false
  const parsed = parseModelConfig(model.value.config)
  if (!parsed || typeof parsed !== 'object') return false
  const legacyFields = [
    'n_gpu_layers', 'main_gpu', 'tensor_split', 'ctx_size', 'batch_size', 'ubatch_size',
    'no_mmap', 'mlock', 'low_vram', 'threads', 'threads_batch', 'parallel',
    'flash_attn', 'cont_batching', 'no_kv_offload', 'logits_all', 'embedding',
    'cache_type_k', 'cache_type_v', 'moe_offload_pattern', 'moe_offload_custom',
    'n_predict', 'temp', 'repeat_penalty',
    'typical_p', 'tfs_z',
    'mirostat', 'mirostat_tau', 'mirostat_eta', 'seed', 'stop', 'grammar',
    'json_schema', 'jinja', 'rope_freq_base', 'rope_freq_scale',
    'yarn_ext_factor', 'yarn_attn_factor', 'rope_scaling', 'yaml', 'customArgs'
  ]
  return legacyFields.some(field => parsed[field] !== undefined && parsed[field] !== null && parsed[field] !== '')
})

</script>

<style scoped>
.model-config {
  min-height: 100vh;
  background: var(--bg-primary);
  overflow-x: hidden;
}

.config-layout {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: var(--spacing-lg);
  max-width: 1400px;
  margin: 0 auto;
  padding: var(--spacing-lg);
  box-sizing: border-box;
  align-items: start;
}

.config-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  overflow-x: hidden;
}

.config-tabs-wrapper {
  width: 100%;
  margin-top: var(--spacing-sm);
}

.config-tabs {
  width: 100%;
}

.config-tabs :deep(.p-tabview-nav) {
  background: transparent;
  border-bottom: 2px solid var(--border-primary);
  border-radius: 0;
  padding: 0;
  margin-bottom: var(--spacing-md);
  gap: 0;
  display: flex;
  align-items: center;
}

.config-tabs :deep(.p-tabview-nav li) {
  margin-right: var(--spacing-sm);
}

.config-tabs :deep(.p-tabview-nav li:last-child) {
  margin-right: 0;
}

.config-tabs :deep(.p-tabview-nav li .p-tabview-nav-link) {
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-md) var(--radius-md) 0 0;
  transition: all var(--transition-normal);
  color: var(--text-secondary);
  border: none;
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
  background: transparent;
  font-weight: 500;
}

.config-tabs :deep(.p-tabview-nav li.p-highlight .p-tabview-nav-link) {
  background: transparent;
  color: var(--accent-cyan);
  font-weight: 600;
  border-bottom-color: var(--accent-cyan);
}

.config-tabs :deep(.p-tabview-nav li .p-tabview-nav-link:hover) {
  background: var(--bg-surface);
  color: var(--text-primary);
  border-bottom-color: var(--text-secondary);
}

.config-tabs :deep(.p-tabview-nav li.p-highlight .p-tabview-nav-link:hover) {
  border-bottom-color: var(--accent-cyan);
  color: var(--accent-cyan);
}

.config-tabs :deep(.p-tabview-panels) {
  padding: 0;
  background: transparent;
  border: none;
}

.tab-content {
  padding: var(--spacing-md);
  min-height: 200px;
  background: transparent;
}

.tab-section {
  margin-bottom: var(--spacing-md);
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-md);
}

.tab-section:last-child {
  margin-bottom: 0;
}

.tab-section-title {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  margin-bottom: var(--spacing-md);
  padding-bottom: var(--spacing-xs);
  border-bottom: 1px solid var(--border-primary);
  color: var(--text-primary);
  font-size: 1rem;
  font-weight: 600;
  grid-column: 1 / -1;
}

.tab-section-title i {
  font-size: 1.2rem;
  color: var(--accent-cyan);
}

.config-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-md);
}

.model-info {
  flex: 1;
}

.model-tag.tag-architecture {
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
  border: 1px solid rgba(59, 130, 246, 0.2);
}

.model-tag.tag-layers {
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
  border: 1px solid rgba(34, 211, 238, 0.2);
}

/* Configuration Warnings */
.config-warnings {
  margin-top: var(--spacing-lg);
  padding: var(--spacing-md);
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-md);
}

.warnings-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-md);
}

.warning-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-md);
  font-size: 0.9rem;
  transition: all var(--transition-normal);
}

.warning-item.warning {
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  color: var(--status-warning);
}

.warning-item.error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: var(--status-error);
}

.warning-item i {
  font-size: 1rem;
  flex-shrink: 0;
}

.warning-item.warning i {
  color: var(--status-warning);
}

.warning-item.error i {
  color: var(--status-error);
}

.meta-item {
  padding: var(--spacing-xs) var(--spacing-sm);
  background: var(--bg-surface);
  border-radius: var(--radius-sm);
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.header-actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-sm);
  align-items: center;
}

.preset-buttons {
  display: flex;
  gap: var(--spacing-xs);
}

.smart-auto-settings-button {
  margin-left: var(--spacing-xs);
}

/* Menu styling for Smart Auto */
:deep(.p-menu) {
  padding: 0.75rem !important;
  min-width: 220px;
  background: var(--surface-ground) !important;
  border: 1px solid var(--surface-border) !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

:deep(.p-menu .p-menuitem) {
  margin: 0.375rem 0 !important;
}

:deep(.p-menu .p-menuitem-link) {
  padding: 0.875rem 1.125rem !important;
  border-radius: 0.5rem !important;
  transition: all 0.2s ease !important;
  color: var(--text-color) !important;
  background: transparent !important;
}

:deep(.p-menu .p-menuitem-link .p-menuitem-text) {
  color: var(--text-color) !important;
  font-weight: 500 !important;
}

:deep(.p-menu .p-menuitem-link .p-menuitem-icon) {
  color: var(--text-color-secondary) !important;
  margin-right: 0.75rem !important;
}

:deep(.p-menu .p-menuitem-link:hover) {
  background: var(--primary-color) !important;
}

:deep(.p-menu .p-menuitem-link:hover .p-menuitem-text) {
  color: white !important;
}

:deep(.p-menu .p-menuitem-link:hover .p-menuitem-icon) {
  color: white !important;
}

:deep(.p-menu .p-menuitem-separator) {
  margin: 0.625rem 0 !important;
  border-top: 1px solid var(--surface-border) !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected),
:deep(.p-menu .p-menuitem-link.active) {
  background: rgba(59, 130, 246, 0.15) !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected .p-menuitem-text),
:deep(.p-menu .p-menuitem-link.active .p-menuitem-text) {
  color: var(--primary-color) !important;
  font-weight: 600 !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected .p-menuitem-icon),
:deep(.p-menu .p-menuitem-link.active .p-menuitem-icon) {
  color: var(--primary-color) !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected:hover),
:deep(.p-menu .p-menuitem-link.active:hover) {
  background: var(--primary-color) !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected:hover .p-menuitem-text),
:deep(.p-menu .p-menuitem-link.active:hover .p-menuitem-text) {
  color: white !important;
}

:deep(.p-menu .p-menuitem-link.menu-item-selected:hover .p-menuitem-icon),
:deep(.p-menu .p-menuitem-link.active:hover .p-menuitem-icon) {
  color: white !important;
}

/* Badge styling for Smart Auto button */
:deep(.p-button .p-badge) {
  padding: 0.4rem 0.7rem !important;
  margin-left: 0.5rem !important;
  border-radius: 0.5rem !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  line-height: 1 !important;
  min-width: auto !important;
  background: rgba(255, 255, 255, 0.35) !important;
  color: rgba(0, 0, 0, 0.9) !important;
  border: 1px solid rgba(0, 0, 0, 0.15) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

.usage-mode-badge {
  padding: 0.4rem 0.7rem !important;
  margin-left: 0.5rem !important;
  border-radius: 0.5rem !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  line-height: 1 !important;
  min-width: auto !important;
  background: rgba(255, 255, 255, 0.35) !important;
  color: rgba(0, 0, 0, 0.9) !important;
  border: 1px solid rgba(0, 0, 0, 0.15) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}


.action-buttons {
  display: flex;
  gap: var(--spacing-xs);
  align-items: center;
}

/* Quick Start Button - Now in header, no separate styles needed */

/* Quick Start Modal Styles */
:deep(.quick-start-modal) {
  max-width: 800px;
  width: 90vw;
}

:deep(.quick-start-modal .p-dialog-header) {
  padding: var(--spacing-xl);
  border-bottom: 1px solid var(--border-primary);
}

:deep(.quick-start-modal .p-dialog-content) {
  padding: var(--spacing-xl);
  max-height: 70vh;
  overflow-y: auto;
}

:deep(.quick-start-modal .p-dialog-footer) {
  padding: var(--spacing-lg) var(--spacing-xl);
  border-top: 1px solid var(--border-primary);
}

.quick-start-modal-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-lg);
}

.quick-start-icon {
  font-size: 3rem;
  flex-shrink: 0;
  line-height: 1;
}

.quick-start-modal-header h3 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--text-primary);
}

.quick-start-modal-header p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 1rem;
}

.quick-start-content {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: var(--spacing-xl);
}

.preset-cards {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.preset-card {
  background: var(--bg-surface);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  cursor: pointer;
  transition: all var(--transition-normal);
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.preset-card:hover {
  border-color: var(--accent-cyan);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.preset-card.wizard-card {
  border-color: rgba(34, 211, 238, 0.5);
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.05), rgba(59, 130, 246, 0.05));
}

.preset-card.wizard-card:hover {
  border-color: var(--accent-cyan);
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.1), rgba(59, 130, 246, 0.1));
}

.preset-icon {
  font-size: 2rem;
  flex-shrink: 0;
}

.preset-info h4 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.preset-info p {
  margin: 0;
  font-size: 0.875rem;
  color: var(--text-secondary);
  line-height: 1.4;
}

.smart-auto-section {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
}

.smart-auto-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-sm);
}

.smart-auto-header i {
  font-size: 1.5rem;
  color: var(--accent-primary);
}

.smart-auto-header h4 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.smart-auto-description {
  margin: 0 0 var(--spacing-lg) 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.5;
}

.usage-mode-selector {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-lg);
}

.smart-auto-controls {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.smart-auto-controls .control-row {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.smart-auto-controls .control-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-sm);
}

.smart-auto-controls .control-label label {
  font-weight: 600;
  color: var(--text-primary);
}

.smart-auto-controls .control-hint {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.smart-auto-controls .control-grid {
  display: grid;
  gap: var(--spacing-md);
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}

.smart-auto-controls .control-field {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.smart-auto-controls .control-field label {
  font-weight: 600;
  color: var(--text-primary);
}

.smart-auto-controls .control-field small {
  font-weight: normal;
  color: var(--text-tertiary);
}

.smart-auto-controls .debug-toggle {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.smart-auto-controls .debug-toggle label {
  font-weight: 500;
  color: var(--text-secondary);
}

.topology-summary-card {
  margin: var(--spacing-lg) 0;
  padding: var(--spacing-lg);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-surface);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.topology-summary-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--spacing-lg);
}

.topology-summary-header h4 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.topology-summary-header p {
  margin: var(--spacing-xs) 0 0 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.topology-summary-metric {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: var(--spacing-xs);
}

.metric-label {
  font-size: 0.8rem;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metric-value {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.topology-summary-grid {
  display: grid;
  gap: var(--spacing-md);
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
}

.topology-warnings {
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-md);
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.2);
  color: var(--text-warning);
}

.topology-warnings ul {
  margin: 0;
  padding-left: var(--spacing-lg);
}

.topology-groups {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.topology-group-card {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-md);
  background: var(--bg-contrast);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.group-header {
  display: flex;
  justify-content: space-between;
  gap: var(--spacing-md);
  align-items: baseline;
}

.group-header h5 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.group-header p {
  margin: var(--spacing-xs) 0 0 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.group-penalty {
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  background: rgba(239, 68, 68, 0.15);
  color: var(--status-error);
  font-size: 0.85rem;
  font-weight: 600;
}

.group-device-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.group-device-table th,
.group-device-table td {
  padding: var(--spacing-xs) var(--spacing-sm);
  text-align: left;
  border-bottom: 1px solid var(--border-primary);
}

.group-device-table th {
  font-weight: 600;
  color: var(--text-secondary);
  background: rgba(255, 255, 255, 0.04);
}

.group-device-table td {
  color: var(--text-primary);
}

.group-warnings {
  background: rgba(239, 68, 68, 0.1);
  border-color: rgba(239, 68, 68, 0.2);
  color: var(--status-error);
}

.radio-option {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-normal);
  background: transparent;
}

.radio-option:hover {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.05);
}

.radio-option.active {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.1);
}

.radio-option i {
  font-size: 1.5rem;
  color: var(--accent-cyan);
  flex-shrink: 0;
}

.radio-option strong {
  display: block;
  margin-bottom: var(--spacing-xs);
  font-size: 1rem;
  color: var(--text-primary);
}

.radio-option small {
  display: block;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.smart-auto-button {
  width: 100%;
}

.flash-attention-warning {
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  margin-bottom: var(--spacing-md);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-md);
  color: var(--status-error);
}

.flash-attention-warning i {
  font-size: 1.5rem;
  flex-shrink: 0;
  margin-top: 2px;
}

.warning-content strong {
  display: block;
  margin-bottom: var(--spacing-xs);
  font-weight: 600;
}

.warning-content p {
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.5;
}

.expert-info {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

/* Configuration Grid */

.info-banner {
  background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
  color: white;
  padding: 1rem 1.5rem;
  border-radius: 0.75rem;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.info-banner i {
  font-size: 1.5rem;
  flex-shrink: 0;
  margin-top: 0.1rem;
  opacity: 0.9;
}

.banner-content {
  flex: 1;
}

.banner-content strong {
  display: block;
  font-size: 1.1rem;
  margin-bottom: 0.25rem;
}

.banner-content p {
  margin: 0;
  opacity: 0.95;
  line-height: 1.5;
}

.info-message {
  background: rgba(59, 130, 246, 0.1);
  border: 1px solid rgba(59, 130, 246, 0.3);
  color: var(--text-primary);
  padding: 0.75rem 1rem;
  border-radius: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.info-message i {
  color: #3b82f6;
  font-size: 1.25rem;
}

.config-search-bar {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  width: 250px;
  transition: width var(--transition-normal);
  margin-bottom: var(--spacing-md);
}

.config-search-bar.search-focused {
  width: 350px;
}

.search-section {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  flex: 0 0 auto;
  width: 200px;
  transition: width var(--transition-normal);
}

.search-section.search-focused {
  width: 300px;
}

.config-search-input {
  width: 100%;
  padding-left: 2rem;
  transition: all var(--transition-normal);
}

.config-search-bar .p-input-icon-left {
  position: relative;
  width: 100%;
}

.config-search-bar .p-input-icon-left i {
  position: absolute;
  left: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-secondary);
  z-index: 1;
  font-size: 0.875rem;
}

.section-controls {
  display: flex;
  gap: var(--spacing-sm);
  flex-shrink: 0;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: var(--spacing-xl);
  margin-top: var(--spacing-lg);
}

.config-section {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
}

.config-section > summary {
  cursor: pointer;
  padding: var(--spacing-lg);
  list-style: none;
}

.config-section > summary::-webkit-details-marker {
  display: none;
}

.config-section[open] > summary {
  border-bottom: 1px solid var(--border-primary);
}

.config-section:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0;
  color: var(--text-primary);
  font-size: 1.1rem;
  font-weight: 600;
  user-select: none;
}

.title-left {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.section-badge {
  display: inline-block;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.section-badge.essential-badge {
  background: rgba(16, 185, 129, 0.15);
  color: var(--status-success);
}

.section-badge.advanced-badge {
  background: rgba(245, 158, 11, 0.15);
  color: var(--status-warning);
}

.section-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: var(--spacing-lg);
  width: 100%;
  min-width: 0;
  padding: var(--spacing-xl);
}

.config-field {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  min-width: 0;
  width: 100%;
}

.config-field.full-width {
  grid-column: 1 / -1;
}

.config-field label {
  font-weight: 500;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.config-field small {
  color: var(--text-secondary);
  font-size: 0.75rem;
  line-height: 1.3;
}

.inline-validation {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 500;
  margin-top: var(--spacing-xs);
  animation: slideIn 0.2s ease-out;
}

.inline-validation.error {
  background: rgba(239, 68, 68, 0.1);
  color: var(--status-error);
  border: 1px solid rgba(239, 68, 68, 0.2);
}

.inline-validation.warning {
  background: rgba(245, 158, 11, 0.1);
  color: var(--status-warning);
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.inline-validation.success {
  background: rgba(16, 185, 129, 0.1);
  color: var(--status-success);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.inline-validation i {
  font-size: 0.875rem;
  flex-shrink: 0;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Sidebar */
.config-sidebar {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
  position: sticky;
  top: var(--spacing-xl);
  max-height: calc(100vh - 2 * var(--spacing-xl));
  overflow-y: auto;
  align-self: start;
}

.vram-monitor {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-lg);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
}

.vram-monitor:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}

.monitor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.monitor-header h3 {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin: 0;
  color: var(--text-primary);
  font-size: 1.1rem;
}

.monitor-meta {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.mode-badge {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: 2px 6px;
  font-weight: 500;
  color: var(--text-primary);
}

.ram-snapshot {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  padding: 2px 6px;
  font-weight: 500;
  color: var(--text-primary);
}

.monitor-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.vram-summary {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.vram-total {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.total-label {
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.total-value {
  font-weight: 700;
  font-size: 1.1rem;
}

.total-value.success {
  color: var(--status-success);
}

.total-value.warning {
  color: var(--status-warning);
}

.vram-progress {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.progress-text {
  font-size: 0.75rem;
  color: var(--text-secondary);
  text-align: center;
}

.vram-breakdown {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.breakdown-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.item-label {
  color: var(--text-secondary);
}

.item-value {
  color: var(--text-primary);
  font-weight: 500;
}

.vram-warning {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: var(--radius-md);
  color: var(--status-warning);
  font-size: 0.875rem;
}

.nvlink-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.3);
  border-radius: var(--radius-md);
  color: var(--accent-cyan);
  font-size: 0.875rem;
}

.monitor-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xl);
  color: var(--text-muted);
  text-align: center;
}

.monitor-empty i {
  font-size: 2rem;
  color: var(--text-muted);
}

.monitor-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
  text-align: center;
}

.monitor-loading i {
  font-size: 2rem;
  color: var(--accent-primary);
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}

.cpu-mode-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.cpu-mode-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-weight: 600;
  color: var(--text-primary);
  font-size: 1rem;
}

.cpu-threads-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.threads-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.threads-label {
  color: var(--text-secondary);
}

.threads-value {
  color: var(--text-primary);
  font-weight: 500;
}

.cpu-performance-tip {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.3);
  border-radius: var(--radius-md);
  color: var(--accent-cyan);
  font-size: 0.875rem;
}

/* RAM Monitor Styles */
.ram-monitor {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-lg);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  margin-bottom: var(--spacing-lg);
}

.ram-monitor:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}

.ram-summary {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.ram-total {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.ram-progress {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.ram-breakdown {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-lg);
}

.ram-warning {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-md);
  color: var(--accent-red);
  font-size: 0.875rem;
}

/* GPU-specific styles for real-time monitoring */
.gpu-item {
  padding: var(--spacing-sm);
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
  margin-bottom: var(--spacing-sm);
}

.gpu-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-xs);
}

.gpu-name {
  font-weight: 500;
  color: var(--text-primary);
  font-size: 0.875rem;
}

.gpu-temp {
  color: var(--text-secondary);
  font-size: 0.75rem;
  background: var(--bg-surface);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
}

.gpu-memory {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-xs);
}

.memory-label {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.memory-value {
  color: var(--text-primary);
  font-size: 0.875rem;
  font-weight: 500;
}

.memory-percent {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.gpu-utilization {
  display: flex;
  gap: var(--spacing-sm);
  font-size: 0.75rem;
}

.util-label {
  color: var(--text-secondary);
}

.util-value {
  color: var(--text-primary);
  font-weight: 500;
}

/* Responsive */
@media (max-width: 1200px) {
  .config-grid {
    grid-template-columns: 1fr;
  }

  .section-grid {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }
}

@media (max-width: 768px) {
  .config-layout {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  .config-sidebar {
    order: -1;
  }

  /* Tablet: Stack memory dashboard */
  .memory-dashboard {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  /* Tablet: Adjust quick start layout */
  .quick-start-content {
    grid-template-columns: 1fr;
    gap: var(--spacing-xl);
  }

  /* Tablet: Full width preset cards */
  .preset-cards {
    width: 100%;
  }

  /* Tablet: Adjust search section */
  .search-section {
    max-width: 100%;
  }

  /* Tablet: 2 column grid for config fields */
  .tab-section {
    grid-template-columns: repeat(2, 1fr);
  }

  .config-search-bar {
    width: 100%;
  }
}

@media (max-width: 600px) {
  .config-layout {
    padding: var(--spacing-md);
  }

  .config-header {
    flex-direction: column;
    align-items: stretch;
    gap: var(--spacing-md);
  }

  .header-actions {
    flex-wrap: wrap;
    justify-content: flex-start;
    gap: var(--spacing-sm);
  }

  .action-buttons {
    flex-wrap: wrap;
  }

  .config-grid {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  /* Mobile: Single column grid for config fields */
  .tab-section {
    grid-template-columns: 1fr;
  }

  .config-search-bar {
    width: 100%;
  }

  /* Memory Dashboard - Stack on mobile */
  .memory-dashboard {
    grid-template-columns: 1fr;
    gap: var(--spacing-md);
  }

  /* Quick Start Modal - Full width on mobile */
  :deep(.quick-start-modal) {
    width: 95vw;
    max-width: none;
  }
  
  :deep(.quick-start-modal .p-dialog-content) {
    max-height: 85vh;
    padding: var(--spacing-lg);
  }
  
  .quick-start-content {
    padding: var(--spacing-lg);
  }

  .quick-start-content {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  .preset-cards {
    flex-direction: column;
  }

  .preset-card {
    min-height: 80px;
    padding: var(--spacing-md);
  }

  .preset-icon {
    font-size: 1.5rem;
  }

  .preset-info h4 {
    font-size: 1rem;
  }

  .preset-info p {
    font-size: 0.85rem;
  }

  /* Smart Auto Section - Full width */
  .smart-auto-section {
    width: 100%;
  }

  .usage-mode-selector {
    flex-direction: column;
    gap: var(--spacing-sm);
  }

  .usage-mode-selector .radio-option {
    min-height: 60px;
    padding: var(--spacing-md);
  }

  .smart-auto-controls .control-grid {
    grid-template-columns: 1fr;
  }

  .smart-auto-controls .control-label {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--spacing-xs);
  }

  /* Config Controls - Full width search */
  .config-controls {
    padding: var(--spacing-md);
  }

  .controls-row {
    flex-direction: column;
    gap: var(--spacing-md);
  }

  .search-section {
    width: 100%;
  }

  .config-search-input {
    width: 100%;
  }

  .section-controls {
    width: 100%;
    justify-content: space-between;
  }

  /* Performance Card - Stack metrics */
  .performance-metrics {
    grid-template-columns: 1fr;
    gap: var(--spacing-md);
  }

  .metric-item {
    padding: var(--spacing-md);
  }

  /* Touch targets - Minimum 44x44px */
  .preset-card,
  .radio-option,
  button,
  .p-button {
    min-height: 44px;
    min-width: 44px;
  }

  /* Larger spacing for touch */
  .config-field {
    gap: var(--spacing-md);
  }

  /* Full width inputs on mobile */
  input[type="number"],
  input[type="text"],
  textarea,
  .p-inputnumber-input,
  .p-inputtext {
    width: 100%;
    min-width: 0;
  }

  /* Adjust section spacing */
  .config-section {
    margin-bottom: var(--spacing-lg);
  }

  /* Collapsible sections - Larger tap area */
  .config-section summary {
    min-height: 48px;
    padding: var(--spacing-md);
  }

  .section-grid {
    grid-template-columns: 1fr;
  }

  .model-meta {
    flex-wrap: wrap;
  }
}

/* Enhanced Focus Styles for Accessibility */
.preset-card:focus {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
  box-shadow: 0 0 0 3px var(--focus-ring), var(--shadow-md);
}

.preset-card:focus-visible {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
}

.radio-option:focus {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
  box-shadow: 0 0 0 3px var(--focus-ring);
}

.radio-option:focus-visible {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
}

.config-section summary:focus {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
}

.config-section summary:focus-visible {
  outline: 3px solid var(--accent-cyan);
  outline-offset: 2px;
}

/* Skip to main content link for keyboard navigation */
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: var(--accent-cyan);
  color: white;
  padding: var(--spacing-sm) var(--spacing-md);
  text-decoration: none;
  z-index: 1000;
  border-radius: var(--radius-md);
}

.skip-link:focus {
  top: var(--spacing-md);
  outline: 3px solid var(--accent-blue);
  outline-offset: 2px;
}

.stacked-bar {
  position: relative;
  width: 100%;
  height: 10px;
  background: var(--bg-secondary);
  border-radius: 5px;
  overflow: hidden;
}

.stacked-bar .bar-current {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  background: var(--accent-red);
}

.stacked-bar .bar-additional {
  position: absolute;
  top: 0;
  bottom: 0;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
  opacity: 0.8;
  transform-origin: left;
  /* left position is set via inline style dynamically */
}

/* Use inline styles for widths; classes control colors */
.stacked-bar.success {
  box-shadow: inset 0 0 0 1px rgba(16, 185, 129, 0.2);
}

.stacked-bar.warning {
  box-shadow: inset 0 0 0 1px rgba(234, 179, 8, 0.3);
}

/* New Memory Dashboard Styles */
.memory-dashboard {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(400px, 100%), 1fr));
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-xl);
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  min-width: 0;
  overflow-x: visible;
}

@media (max-width: 1400px) {
  .memory-dashboard {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 900px) {
  .memory-dashboard {
    grid-template-columns: 1fr;
    min-width: 0;
  }
}

@media (max-width: 480px) {
  .memory-dashboard {
    grid-template-columns: 1fr;
    gap: var(--spacing-md);
  }
}

.memory-status-card {
  background: var(--gradient-card);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  min-width: 0;
  max-width: 100%;
  box-sizing: border-box;
  word-wrap: break-word;
}

.memory-status-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--border-primary);
  transition: all var(--transition-normal);
}

.memory-status-card.status-good::before {
  background: var(--gradient-success);
}

.memory-status-card.status-warning::before {
  background: var(--gradient-warning);
}

.memory-status-card.status-critical::before {
  background: var(--gradient-error);
}

.memory-card-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.memory-status-icon {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-lg);
  font-size: 1.5rem;
  flex-shrink: 0;
}

.memory-status-card.status-good .memory-status-icon {
  background: rgba(16, 185, 129, 0.15);
  color: var(--status-success);
}

.memory-status-card.status-warning .memory-status-icon {
  background: rgba(245, 158, 11, 0.15);
  color: var(--status-warning);
}

.memory-status-card.status-critical .memory-status-icon {
  background: rgba(239, 68, 68, 0.15);
  color: var(--status-error);
}

.memory-card-title {
  flex: 1;
}

.memory-card-title h4 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
}

.memory-status-badge {
  display: inline-block;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.875rem;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.memory-status-card.status-good .memory-status-badge {
  background: rgba(16, 185, 129, 0.15);
  color: var(--status-success);
}

.memory-status-card.status-warning .memory-status-badge {
  background: rgba(245, 158, 11, 0.15);
  color: var(--status-warning);
}

.memory-status-card.status-critical .memory-status-badge {
  background: rgba(239, 68, 68, 0.15);
  color: var(--status-error);
}

.memory-status-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.memory-usage-display {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
}

.usage-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.9rem;
}

.usage-item.total {
  font-weight: 600;
  padding-top: var(--spacing-sm);
  border-top: 1px solid var(--border-primary);
  font-size: 1rem;
}

.usage-label {
  color: var(--text-secondary);
  min-width: 80px;
}

.usage-value {
  color: var(--text-primary);
  font-weight: 500;
  flex: 1;
}

.usage-item.total .usage-value {
  font-weight: 700;
  color: var(--accent-cyan);
}

.usage-fraction {
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.memory-progress-bar {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.progress-label {
  font-size: 0.875rem;
  color: var(--text-secondary);
  text-align: center;
}

.memory-message {
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  font-size: 0.9rem;
  line-height: 1.5;
  font-weight: 500;
}

.memory-message.status-good {
  background: rgba(16, 185, 129, 0.1);
  color: var(--status-success);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.memory-message.status-warning {
  background: rgba(245, 158, 11, 0.1);
  color: var(--status-warning);
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.memory-message.status-critical {
  background: rgba(239, 68, 68, 0.1);
  color: var(--status-error);
  border: 1px solid rgba(239, 68, 68, 0.2);
}

.memory-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-md);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
}

.memory-loading i {
  font-size: 2rem;
  color: var(--accent-primary);
  animation: spin 1s linear infinite;
}

.advanced-section {
  grid-column: 1 / -1;
  padding: var(--spacing-sm) 0;
}

.advanced-section summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--spacing-sm);
}

.advanced-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--spacing-md);
}
</style>