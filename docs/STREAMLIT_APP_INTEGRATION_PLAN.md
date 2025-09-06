# Streamlit App Integration Plan

## Current Issues
- Import problems: App imports from missing `modules.*` instead of `music_anomalizer.*`
- Missing `Home.py` main entry point
- Hardcoded relative paths and pickle dependencies
- Poor integration with main Python package

## Phase 1: Fix Import Issues & Structure ✅
- [x] Fix imports to use `music_anomalizer` package
- [x] Create proper `Home.py` main entry point  
- [x] Update path resolution for package structure
- [x] **Option A Implementation Complete:**
  - [x] Update dataset paths to read from config dynamically
  - [x] Use config thresholds as defaults with UI override
  - [x] Fix relative path resolution for embeddings and checkpoints
  - [x] Add config selection dropdown for experiment switching

## Phase 2: Integration Testing
- [ ] Test audio upload and analysis functionality
- [ ] Verify model loading and inference pipeline
- [ ] Test visualization and PCA plotting features
- [ ] Ensure proper error handling

## Phase 3: End-to-End Validation
- [ ] Test complete user workflow (upload → analyze → visualize)
- [ ] Validate against package test suite
- [ ] Check model checkpoints and data dependencies
- [ ] Performance and reliability testing

## Phase 4: Optimization & Documentation
- [ ] Add proper logging integration
- [ ] Optimize caching and performance
- [ ] Create usage documentation
- [ ] Add health checks

## Key Testing Areas
1. **Model Loading**: AnomalyDetector and Wav2Embedding integration
2. **Data Pipeline**: Audio preprocessing and embedding extraction
3. **UI Components**: Streamlit interface validation
4. **File Dependencies**: Required pickle files and checkpoints
5. **Cross-Integration**: Consistent configs/models with main package