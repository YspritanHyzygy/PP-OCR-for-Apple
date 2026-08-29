export async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(await detail(response));
  return response.json();
}

export async function analyzeImage(file, selection) {
  const body = new FormData();
  body.append("image", file);
  body.append("detector", selection.detector);
  body.append("classifier", selection.classifier);
  body.append("recognizer", selection.recognizer);
  body.append("threshold", String(selection.threshold));
  const response = await fetch("/api/analyze", { method: "POST", body });
  if (!response.ok) throw new Error(await detail(response));
  return response.json();
}

export async function runTranslationPipeline(file, selection) {
  const body = new FormData();
  body.append("image", file);
  body.append("ocr_provider", selection.ocrProvider);
  body.append("translation_provider", selection.translationProvider);
  body.append("reconstructor", selection.reconstructor);
  body.append("detector", selection.detector);
  body.append("classifier", selection.classifier);
  body.append("recognizer", selection.recognizer);
  body.append("threshold", String(selection.threshold));
  body.append("source_language", selection.sourceLanguage);
  body.append("target_language", selection.targetLanguage);
  body.append("cloud_feature", selection.cloudFeature);
  body.append("cloud_grouping", selection.cloudGrouping);
  const response = await fetch("/api/pipeline", { method: "POST", body });
  if (!response.ok) throw new Error(await detail(response));
  return response.json();
}

export async function runCloudOCR(file, sourceLanguage = "auto", cloudFeature = "DOCUMENT_TEXT_DETECTION", cloudGrouping = "google-paragraph") {
  const body = new FormData();
  body.append("image", file);
  body.append("source_language", sourceLanguage);
  body.append("cloud_feature", cloudFeature);
  body.append("cloud_grouping", cloudGrouping);
  const response = await fetch("/api/cloud-ocr", { method: "POST", body });
  if (!response.ok) throw new Error(await detail(response));
  return response.json();
}

async function detail(response) {
  try {
    const payload = await response.json();
    return payload.detail || `请求失败（${response.status}）`;
  } catch {
    return `请求失败（${response.status}）`;
  }
}
