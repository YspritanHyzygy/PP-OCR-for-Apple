import {
  Check,
  Cloud,
  Download,
  FileJson,
  FolderOpen,
  Hand,
  ImagePlus,
  Layers3,
  LoaderCircle,
  Minus,
  MousePointer2,
  Play,
  Plus,
  Search,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { analyzeImage, getJSON, runCloudOCR as runCloudOCRRequest, runTranslationPipeline } from "./api";

const REGION_COLORS = ["#2563eb", "#16a34a", "#7c3aed", "#ea580c", "#d9a400", "#ef4444"];
const TABS = ["检测", "方向", "识别", "耗时", "JSON"];

function formatMs(value) {
  return `${Math.round(value || 0)} ms`;
}

function scriptName(value) {
  return {
    latin: "拉丁",
    han: "汉字",
    kana: "日文假名",
    hangul: "韩文",
    other: "其他",
    unknown: "待判断",
  }[value] || value;
}

function layoutName(layout) {
  if (layout.writing_mode === "vertical") {
    return layout.reading_direction === "bottom-to-top" ? "竖排 · 从下到上" : "竖排 · 从上到下";
  }
  if (layout.writing_mode === "horizontal") {
    return layout.reading_direction === "right-to-left" ? "横排 · 从右到左" : "横排 · 从左到右";
  }
  return "待判断";
}

export function App() {
  const [health, setHealth] = useState({ status: "connecting", runtime: "ONNX Runtime" });
  const [catalog, setCatalog] = useState({ detectors: [], classifiers: [], recognizers: [] });
  const [selection, setSelection] = useState({
    detector: "reference-geometry",
    classifier: "geometry-fallback",
    recognizer: "none",
    threshold: 0.2,
  });
  const [selectionB, setSelectionB] = useState({
    detector: "reference-geometry",
    classifier: "geometry-fallback",
    recognizer: "none",
    threshold: 0.55,
  });
  const [sessions, setSessions] = useState([]);
  const [activeId, setActiveId] = useState("sample");
  const [result, setResult] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [pipelineImageURL, setPipelineImageURL] = useState("");
  const [integrations, setIntegrations] = useState({
    cloud: { ocr: { available: false }, translation: { available: false } },
    migan: { available: false },
  });
  const [pipelineSelection, setPipelineSelection] = useState({
    ocrProvider: "local",
    translationProvider: "google-cloud-translation",
    reconstructor: "migan-512",
    detector: "auto",
    classifier: "geometry-fallback",
    recognizer: "auto",
    threshold: 0.35,
    sourceLanguage: "auto",
    targetLanguage: "zh-Hans",
    cloudFeature: "DOCUMENT_TEXT_DETECTION",
    cloudGrouping: "google-paragraph",
  });
  const [pipelineBusy, setPipelineBusy] = useState(false);
  const [cloudOCRBusy, setCloudOCRBusy] = useState(false);
  const [imageURL, setImageURL] = useState("/api/sample-image");
  const [selectedRegionID, setSelectedRegionID] = useState(6);
  const [activeTab, setActiveTab] = useState("检测");
  const [zoom, setZoom] = useState(0.58);
  const [tool, setTool] = useState("inspect");
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [displayMode, setDisplayMode] = useState("overlay");
  const [compare, setCompare] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const inputRef = useRef(null);
  const overlayLinkRef = useRef(null);
  const dragRef = useRef(null);
  const resultRequestRef = useRef(0);
  const [overlayDownload, setOverlayDownload] = useState(null);

  useEffect(() => {
    Promise.all([getJSON("/api/health"), getJSON("/api/models"), getJSON("/api/sample"), getJSON("/api/integrations")])
      .then(([nextHealth, nextCatalog, sample, nextIntegrations]) => {
        setHealth(nextHealth);
        setCatalog(nextCatalog);
        setIntegrations(nextIntegrations);
        if (resultRequestRef.current === 0) {
          setResult(sample);
        }
        setPipelineSelection((current) => ({
          ...current,
          ocrProvider: nextIntegrations.cloud?.ocr?.available ? "google-cloud-vision" : "local",
          reconstructor: nextIntegrations.migan?.available ? "migan-512" : "original",
        }));
        const preferred = nextCatalog.detectors.find(
          (option) => option.id === "pp-ocrv6-tiny-det" && option.available,
        );
        if (preferred) {
          setSelection((current) => ({ ...current, detector: preferred.id, threshold: 0.2 }));
        }
        const preferredRecognizer = nextCatalog.recognizers.find(
          (option) => option.id === "verto-auto-recognizer" && option.available,
        ) || nextCatalog.recognizers.find(
          (option) => option.id === "pp-ocrv5-mobile-rec" && option.available,
        );
        if (preferredRecognizer) {
          setSelection((current) => ({ ...current, recognizer: preferredRecognizer.id }));
        }
        const appleVision = nextCatalog.detectors.find(
          (option) => option.id === "apple-vision-ocr" && option.available,
        );
        if (appleVision) {
          setSelectionB((current) => ({
            ...current,
            detector: appleVision.id,
            recognizer: appleVision.id,
          }));
        }
      })
      .catch((nextError) => setError(nextError.message));
  }, []);

  useEffect(() => {
    if (overlayDownload) overlayLinkRef.current?.click();
    return () => {
      if (overlayDownload?.url) URL.revokeObjectURL(overlayDownload.url);
    };
  }, [overlayDownload]);

  useEffect(() => {
    if (pipelineSelection.ocrProvider !== "local") setCompare(false);
  }, [pipelineSelection.ocrProvider]);

  const activeSession = sessions.find((session) => session.id === activeId);
  const selectedRegion = result?.regions.find((region) => region.id === selectedRegionID) || result?.regions[0];
  const displayImageURL = displayMode === "final" && pipelineImageURL ? pipelineImageURL : imageURL;

  async function runAnalysis(session = activeSession) {
    if (!session?.file) {
      setToast("示例夹具已加载；上传照片后可运行实际 detector。");
      return;
    }
    const requestID = ++resultRequestRef.current;
    setBusy(true);
    setError("");
    try {
      const [nextResult, nextComparison] = await Promise.all([
        analyzeImage(session.file, selection),
        compare ? analyzeImage(session.file, selectionB) : Promise.resolve(null),
      ]);
      if (requestID !== resultRequestRef.current) return;
      setResult(nextResult);
      setComparisonResult(nextComparison);
      setSelectedRegionID(nextResult.regions[0]?.id || null);
      setSessions((items) => items.map((item) => (
        item.id === session.id
          ? { ...item, result: nextResult, comparisonResult: nextComparison }
          : item
      )));
    } catch (nextError) {
      setError(nextError.message);
    } finally {
      if (requestID === resultRequestRef.current) setBusy(false);
    }
  }

  async function runPipeline(session = activeSession) {
    if (!session?.file) {
      setToast("请先上传一张照片；示例夹具不能调用云端流程。");
      return;
    }
    const requestID = ++resultRequestRef.current;
    setPipelineBusy(true);
    setError("");
    setPipelineResult(null);
    setPipelineImageURL("");
    try {
      const nextPipelineResult = await runTranslationPipeline(session.file, {
        ...pipelineSelection,
        detector: selection.detector,
        classifier: selection.classifier,
        recognizer: selection.recognizer,
        threshold: selection.threshold,
      });
      if (requestID !== resultRequestRef.current) return;
      const nextImageURL = `data:${nextPipelineResult.final_image_media_type};base64,${nextPipelineResult.final_image_base64}`;
      // The API response is atomic: do not expose a repaired or translated
      // intermediate image before this one complete payload arrives.
      setPipelineResult(nextPipelineResult);
      setPipelineImageURL(nextImageURL);
      setResult(nextPipelineResult.analysis);
      setDisplayMode("final");
      setSelectedRegionID(nextPipelineResult.analysis.regions[0]?.id || null);
      setSessions((items) => items.map((item) => (
        item.id === session.id ? { ...item, pipelineResult: nextPipelineResult } : item
      )));
    } catch (nextError) {
      setError(nextError.message);
    } finally {
      if (requestID === resultRequestRef.current) setPipelineBusy(false);
    }
  }

  async function runCloudOCR(session = activeSession) {
    if (!session?.file) {
      setToast("请先上传一张照片；示例夹具不能调用云端流程。");
      return;
    }
    const requestID = ++resultRequestRef.current;
    setCloudOCRBusy(true);
    setError("");
    setResult(null);
    setComparisonResult(null);
    setPipelineResult(null);
    setPipelineImageURL("");
    setDisplayMode("overlay");
    try {
      const nextResult = await runCloudOCRRequest(
        session.file,
        pipelineSelection.sourceLanguage,
        pipelineSelection.cloudFeature,
        pipelineSelection.cloudGrouping,
      );
      if (requestID !== resultRequestRef.current) return;
      setResult(nextResult);
      setSelectedRegionID(nextResult.regions[0]?.id || null);
      setSessions((items) => items.map((item) => (
        item.id === session.id ? { ...item, result: nextResult, comparisonResult: null, pipelineResult: null } : item
      )));
    } catch (nextError) {
      setError(nextError.message);
    } finally {
      if (requestID === resultRequestRef.current) setCloudOCRBusy(false);
    }
  }

  function acceptFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) => file.type.startsWith("image/")).slice(0, 50);
    if (!files.length) return;
    const added = files.map((file) => ({
      id: `${file.name}-${file.lastModified}-${crypto.randomUUID()}`,
      name: file.name,
      size: file.size,
      file,
      url: URL.createObjectURL(file),
      result: null,
      comparisonResult: null,
      pipelineResult: null,
    }));
    setSessions((items) => [...added, ...items]);
    setActiveId(added[0].id);
    setImageURL(added[0].url);
    setResult(added[0].result);
    setComparisonResult(null);
    setPipelineResult(null);
    setPipelineImageURL("");
    setDisplayMode("overlay");
    setSelectedRegionID(null);
  }

  function selectSession(session) {
    resultRequestRef.current += 1;
    setActiveId(session.id);
    setImageURL(session.url);
    setResult(session.result);
    setComparisonResult(session.comparisonResult);
    setPipelineResult(session.pipelineResult || null);
    setPipelineImageURL(session.pipelineResult ? `data:${session.pipelineResult.final_image_media_type};base64,${session.pipelineResult.final_image_base64}` : "");
    setDisplayMode(session.pipelineResult ? "final" : "overlay");
    setSelectedRegionID(session.result?.regions[0]?.id || null);
  }

  function selectSample() {
    resultRequestRef.current += 1;
    setActiveId("sample");
    setImageURL("/api/sample-image");
    getJSON("/api/sample").then((sample) => {
      setResult(sample);
      setComparisonResult(compare ? sample : null);
      setPipelineResult(null);
      setPipelineImageURL("");
      setDisplayMode("overlay");
      setSelectedRegionID(6);
    });
  }

  function notify(message) {
    setToast(message);
    window.setTimeout(() => setToast(""), 2400);
  }

  function resetViewport(nextZoom) {
    setZoom(nextZoom);
    setPan({ x: 0, y: 0 });
  }

  function beginPan(event) {
    if (tool !== "pan" || event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, pan };
    setDragging(true);
  }

  function movePan(event) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    setPan({
      x: drag.pan.x + event.clientX - drag.x,
      y: drag.pan.y + event.clientY - drag.y,
    });
  }

  function endPan(event) {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setDragging(false);
  }

  function zoomWithWheel(event) {
    event.preventDefault();
    setZoom((value) => Math.min(2, Math.max(0.2, value - event.deltaY * 0.001)));
  }

  async function exportOverlay() {
    if (!result) return;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageURL;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = result.width;
    canvas.height = result.height;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0, result.width, result.height);
    context.lineWidth = Math.max(2, result.width / 700);
    context.font = `${Math.max(16, result.width / 65)}px system-ui`;
    result.regions.forEach((region, index) => {
      const color = REGION_COLORS[index % REGION_COLORS.length];
      context.strokeStyle = color;
      context.fillStyle = color;
      context.beginPath();
      region.quad.forEach((point, pointIndex) => {
        if (pointIndex === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.closePath();
      context.stroke();
      context.fillText(String(region.id), region.quad[0].x + 5, region.quad[0].y - 5);
    });
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("无法生成带框图片。");
    setOverlayDownload({
      url: URL.createObjectURL(blob),
      name: `${result.image_name.replace(/\.[^.]+$/, "")}-overlay.png`,
    });
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">Verto OCR Lab</div>
        <div className="connection">
          <span className={`status-dot ${health.status === "connected" ? "connected" : ""}`} />
          本地 · ONNX Runtime · 云端测试可选
          <span className="connection-label">{health.status === "connected" ? "已连接" : "连接中"}</span>
        </div>
        <div className="top-actions">
          <button className="text-button" onClick={() => navigator.clipboard.writeText(health.modelRoot || "PP-OCR-for-Apple").then(() => notify("项目路径已复制"))}>
            <FolderOpen size={17} /> 复制项目路径
          </button>
          <a className={`text-button ${result ? "" : "disabled"}`} href={result ? jsonDataURL(result) : undefined} download={result ? `${result.image_name.replace(/\.[^.]+$/, "")}-ocr-lab.json` : undefined}>
            <FileJson size={17} /> 导出报告
          </a>
        </div>
      </header>

      <aside className="sidebar">
        <div className="sidebar-title-row">
          <h2>测试会话</h2>
          <button className="outline-button" onClick={() => inputRef.current?.click()}><Plus size={15} /> 新建</button>
        </div>
        <button className={`session-card ${activeId === "sample" ? "active" : ""}`} onClick={selectSample}>
          <strong>混合方向菜单样张</strong>
          <span>6 个区域 · 固定夹具</span>
        </button>

        <div className="section-label">批量上传</div>
        <button
          className="upload-zone"
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => { event.preventDefault(); acceptFiles(event.dataTransfer.files); }}
        >
          <UploadCloud size={34} strokeWidth={1.6} />
          <strong>拖入照片或点击上传</strong>
          <span>支持 JPG / PNG / WebP</span>
          <span>单次最多 50 张</span>
        </button>
        <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" multiple hidden onChange={(event) => acceptFiles(event.target.files)} />

        <div className="section-label recent-label">最近图像</div>
        <div className="recent-list">
          {sessions.length === 0 ? <div className="empty-list">上传后的图片只保留在本次会话</div> : null}
          {sessions.map((session) => (
            <button key={session.id} className={`recent-row ${activeId === session.id ? "active" : ""}`} onClick={() => selectSession(session)}>
              <img src={session.url} alt="" />
              <span><strong>{session.name}</strong><small>{Math.round(session.size / 1024)} KB</small></span>
              {session.result ? <Check size={15} className="success-icon" /> : null}
            </button>
          ))}
        </div>

        {pipelineSelection.ocrProvider === "local" ? <div className="compare-panel">
          <div className="compare-row"><span>对比模式</span><button className={`switch ${compare ? "on" : ""}`} onClick={() => setCompare((value) => !value)}><span /></button></div>
          <label>Model B detector</label>
          <select className="compare-select" value={selectionB.detector} onChange={(event) => setSelectionB({ ...selectionB, detector: event.target.value })}>
            {catalog.detectors.map((option) => <option key={option.id} value={option.id} disabled={!option.available}>{option.label}{option.version ? ` · ${option.version}` : ""}</option>)}
          </select>
          <label>Model B recognizer</label>
          <select className="compare-select" value={selectionB.recognizer} onChange={(event) => setSelectionB({ ...selectionB, recognizer: event.target.value })}>
            {catalog.recognizers.map((option) => <option key={option.id} value={option.id} disabled={!option.available}>{option.label}{option.version ? ` · ${option.version}` : ""}</option>)}
          </select>
          <label>Model B 阈值 · {selectionB.threshold.toFixed(2)}</label>
          <input className="compare-threshold" type="range" min="0.1" max="0.8" step="0.05" value={selectionB.threshold} onChange={(event) => setSelectionB({ ...selectionB, threshold: Number(event.target.value) })} />
        </div> : null}
      </aside>

      <main className="workspace">
        <div className="canvas-toolbar">
          <div className="tool-group">
            <button className={`tool ${tool === "pan" ? "active" : ""}`} aria-label="平移" onClick={() => setTool("pan")}><Hand size={17} /></button>
            <button className={`tool ${tool === "inspect" ? "active" : ""}`} aria-label="检查区域" onClick={() => setTool("inspect")}><MousePointer2 size={17} /></button>
          </div>
          <div className="tool-group">
            <button className="tool" aria-label="缩小" onClick={() => setZoom((value) => Math.max(0.2, value - 0.1))}><Minus size={16} /></button>
            <button className="tool" aria-label="放大" onClick={() => setZoom((value) => Math.min(2, value + 0.1))}><Plus size={16} /></button>
            <button className="zoom-value" onClick={() => resetViewport(0.58)}>{Math.round(zoom * 100)}%</button>
          </div>
          <button className="plain-control" onClick={() => resetViewport(0.58)}>适应窗口</button>
          <button className="plain-control" onClick={() => resetViewport(1)}>原始大小</button>
          <div className="toolbar-spacer" />
          <button className={`segmented ${displayMode === "original" ? "active" : ""}`} onClick={() => { setDisplayMode("original"); setShowOverlay(false); }}>原图</button>
          <button className={`segmented ${displayMode === "overlay" ? "active" : ""}`} onClick={() => { setDisplayMode("overlay"); setShowOverlay(true); }}>检测叠加</button>
          <button className={`segmented ${displayMode === "final" ? "active" : ""}`} disabled={!pipelineResult} onClick={() => { setDisplayMode("final"); setShowOverlay(false); }}>最终译图</button>
        </div>

        <div
          className={`canvas-surface tool-${tool} ${dragging ? "dragging" : ""}`}
          onPointerDown={beginPan}
          onPointerMove={movePan}
          onPointerUp={endPan}
          onPointerCancel={endPan}
          onWheel={zoomWithWheel}
        >
          {imageURL ? (
            <div className={`stage-grid ${compare ? "comparing" : ""}`}>
              <ImageStage label={compare ? "Model A" : ""} imageURL={displayImageURL} result={result} showOverlay={showOverlay && displayMode === "overlay"} zoom={zoom} pan={pan} selectedRegion={selectedRegion} onSelect={tool === "inspect" ? setSelectedRegionID : () => {}} />
              {compare ? <ImageStage label="Model B" imageURL={imageURL} result={comparisonResult} showOverlay={showOverlay && displayMode === "overlay"} zoom={zoom} pan={pan} selectedRegion={null} onSelect={tool === "inspect" ? setSelectedRegionID : () => {}} /> : null}
            </div>
          ) : <div className="canvas-empty"><ImagePlus size={42} /><span>上传照片开始分析</span></div>}
          {result ? <div className="image-size">{result.width} × {result.height}</div> : null}
          <div className="canvas-hint">手形工具拖拽平移，滚轮缩放，箭头工具点击框查看详情</div>
        </div>
      </main>

      <aside className="inspector">
        <div className="tabs">{TABS.map((tab) => <button key={tab} className={activeTab === tab ? "active" : ""} onClick={() => setActiveTab(tab)}>{tab}</button>)}</div>
        <div className="inspector-body">
          {activeTab === "检测" ? <DetectionInspector region={selectedRegion} result={result} /> : null}
          {activeTab === "方向" ? <DirectionInspector region={selectedRegion} /> : null}
          {activeTab === "识别" ? <RecognitionInspector region={selectedRegion} result={result} /> : null}
          {activeTab === "耗时" ? <TimingInspector timings={pipelineResult?.timings_ms || result?.timings_ms} /> : null}
          {activeTab === "JSON" ? <pre className="json-view">{JSON.stringify(pipelineResult || result, null, 2)}</pre> : null}

          <PipelinePanel
            integrations={integrations}
            selection={pipelineSelection}
            setSelection={setPipelineSelection}
            busy={pipelineBusy}
            cloudOCRBusy={cloudOCRBusy}
            canRun={Boolean(activeSession?.file)}
            onRun={() => runPipeline()}
            onCloudOCR={() => runCloudOCR()}
            pipelineResult={pipelineResult}
          />

          {pipelineSelection.ocrProvider === "local" ? <div className="model-section">
            <h3>模型配置</h3>
            <ModelSelect label="Detector" value={selection.detector} options={catalog.detectors} onChange={(value) => setSelection({ ...selection, detector: value })} />
            <ModelSelect label="Layout Classifier" value={selection.classifier} options={catalog.classifiers} onChange={(value) => setSelection({ ...selection, classifier: value })} />
            <ModelSelect label="Recognizer" value={selection.recognizer} options={catalog.recognizers} onChange={(value) => setSelection({ ...selection, recognizer: value })} />
            <label className="threshold-row"><span>检测阈值</span><input type="range" min="0.1" max="0.8" step="0.05" value={selection.threshold} onChange={(event) => setSelection({ ...selection, threshold: Number(event.target.value) })} /><output>{selection.threshold.toFixed(2)}</output></label>
            <button className="run-button" onClick={() => runAnalysis()} disabled={busy}>
              {busy ? <LoaderCircle size={17} className="spin" /> : <Play size={17} />} {busy ? "分析中" : "运行分析"}
            </button>
            <button className="secondary-export" onClick={exportOverlay} disabled={!result}><Download size={16} /> 导出带框图片</button>
          </div> : null}
        </div>
      </aside>

      <section className="bottom-panel">
        <TimingRail timings={result?.timings_ms} pipelineTimings={pipelineResult?.timings_ms} />
        <ResultsTable result={result} selectedRegionID={selectedRegion?.id} onSelect={setSelectedRegionID} />
      </section>

      {error ? <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div> : null}
      {toast ? <div className="toast">{toast}</div> : null}
      {overlayDownload ? <a ref={overlayLinkRef} className="generated-download" href={overlayDownload.url} download={overlayDownload.name}>下载带框图片</a> : null}
    </div>
  );
}

function DetectionInspector({ region, result }) {
  if (!region) return <InspectorEmpty />;
  return <>
    <div className="inspector-heading"><h3>选中区域</h3><span className="region-id">ID: {region.id}</span></div>
    <dl className="property-list">
      <dt>坐标（像素）</dt><dd className="coordinates">{region.quad.map((point, index) => <code key={index}>[{Math.round(point.x)}, {Math.round(point.y)}]</code>)}</dd>
      <dt>置信度</dt><dd>{region.confidence.toFixed(3)}</dd>
      <dt>识别置信度</dt><dd>{region.recognition_confidence.toFixed(3)}</dd>
      <dt>翻译块 / 行</dt><dd>{region.group_id ? `G${region.group_id} / L${region.line_index}` : "未连接"}</dd>
      <dt>书写模式</dt><dd className={region.layout.writing_mode === "vertical" ? "accent-red" : ""}>{layoutName(region.layout)}</dd>
      <dt>脚本</dt><dd>{scriptName(region.layout.script_family)}</dd>
      <dt>方向置信度</dt><dd>{region.layout.confidence.toFixed(3)}</dd>
      <dt>模型版本</dt><dd>{result?.detector} · {result?.model_versions?.detector || "unknown"}</dd>
    </dl>
    {result?.warnings?.map((warning) => <div className="warning" key={warning}>{warning}</div>)}
  </>;
}

function DirectionInspector({ region }) {
  if (!region) return <InspectorEmpty />;
  return <div className="focus-panel"><Layers3 size={28} /><strong>{layoutName(region.layout)}</strong><span>{scriptName(region.layout.script_family)} · 置信度 {region.layout.confidence.toFixed(3)}</span><p>方向来自区域自身的几何与内容，不读取设备姿态。</p></div>;
}

function RecognitionInspector({ region, result }) {
  if (!region) return <InspectorEmpty />;
  const group = result?.groups?.find((item) => item.id === region.group_id);
  return <div className="recognition-copy"><label>当前行</label><p>{region.text || "当前 detector-only 路径未运行字符识别。"}</p><span>识别置信度 {region.recognition_confidence.toFixed(3)}</span>{group ? <><label className="group-label">翻译块 G{group.id} · {group.lines.length} 行</label><pre>{group.text}</pre></> : null}</div>;
}

function TimingInspector({ timings }) {
  if (!timings) return <InspectorEmpty />;
  return <dl className="timing-list">{Object.entries(timings).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{formatMs(value)}</dd></div>)}</dl>;
}

function InspectorEmpty() {
  return <div className="inspector-empty"><Search size={24} /><span>选择一个文字区域查看详情</span></div>;
}

function ModelSelect({ label, value, options, onChange }) {
  const selected = options.find((option) => option.id === value);
  return <label className="model-row"><span>{label}</span><span className="model-control"><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option key={option.id} value={option.id} disabled={!option.available}>{option.label}{option.version ? ` · ${option.version}` : ""}{option.available ? "" : " · 未准备"}</option>)}</select>{selected?.detail ? <small>{selected.detail}</small> : null}</span></label>;
}

function PipelinePanel({ integrations, selection, setSelection, busy, cloudOCRBusy, canRun, onRun, onCloudOCR, pipelineResult }) {
  const cloudOCR = integrations.cloud?.ocr;
  const cloudTranslation = integrations.cloud?.translation;
  const migan = integrations.migan;
  const waitingForConfiguration = selection.translationProvider === "google-cloud-translation" && !cloudTranslation?.available;
  const waitingForMIGAN = selection.reconstructor === "migan-512" && !migan?.available;
  return <div className="pipeline-section">
    <div className="pipeline-heading"><div><h3>整套翻译流程</h3><p>全部完成后才显示最终译图</p></div><ShieldCheck size={18} /></div>
    <div className="pipeline-flow"><span>OCR</span><i /> <span>翻译</span><i /> <span>修复</span><i /> <span>发布</span></div>
    <label className="pipeline-row"><span><Cloud size={13} /> OCR</span><select value={selection.ocrProvider} onChange={(event) => setSelection({ ...selection, ocrProvider: event.target.value })}>
      <option value="local">本地 OCR</option>
      <option value="google-cloud-vision" disabled={!cloudOCR?.available}>Google Cloud Vision{cloudOCR?.available ? "" : " · 未配置"}</option>
    </select></label>
    {selection.ocrProvider === "google-cloud-vision" ? <>
      <div className="google-ocr-config">
        <div className="google-ocr-config-title">Google OCR 测试配置</div>
        <label className="pipeline-row"><span>识别类型</span><select value={selection.cloudFeature} onChange={(event) => setSelection({ ...selection, cloudFeature: event.target.value })}><option value="DOCUMENT_TEXT_DETECTION">Document Text Detection · 默认</option><option value="TEXT_DETECTION">Text Detection · 对照</option></select></label>
        <label className="pipeline-row"><span>行分组</span><select value={selection.cloudGrouping} onChange={(event) => setSelection({ ...selection, cloudGrouping: event.target.value })}><option value="google-paragraph">Google 原始 Paragraph · 默认</option><option value="verto-geometry">Verto 几何行连接 · 实验</option></select></label>
      </div>
      <button className="cloud-ocr-test-button" onClick={onCloudOCR} disabled={cloudOCRBusy || !canRun || !cloudOCR?.available}>
        {cloudOCRBusy ? <LoaderCircle size={15} className="spin" /> : <Cloud size={15} />} {cloudOCRBusy ? "Google OCR 测试中" : "仅测试 Google OCR"}
      </button>
      <div className="pipeline-subnote">只调用 Vision 的框选与识别，不翻译、不修复背景。</div>
    </> : null}
    <label className="pipeline-row"><span><LanguagesIcon /> 翻译</span><select value={selection.translationProvider} onChange={(event) => setSelection({ ...selection, translationProvider: event.target.value })}>
      <option value="google-cloud-translation">Google Cloud Translation{cloudTranslation?.available ? "" : " · 未配置"}</option>
      <option value="none">不翻译（只测修复）</option>
    </select></label>
    <label className="pipeline-row"><span><Sparkles size={13} /> 背景修复</span><select value={selection.reconstructor} onChange={(event) => setSelection({ ...selection, reconstructor: event.target.value })}>
      <option value="migan-512">MI-GAN 512 · 整图一次推理{migan?.available ? "" : " · 未配置"}</option>
      <option value="original">不修复（保留原图）</option>
    </select></label>
    <div className="pipeline-pair">
      <label><span>源语言</span><select value={selection.sourceLanguage} onChange={(event) => setSelection({ ...selection, sourceLanguage: event.target.value })}><option value="auto">自动</option><option value="en">English</option><option value="ja">日本語</option><option value="zh-CN">简体中文</option><option value="ko">한국어</option><option value="th">ไทย</option></select></label>
      <label><span>目标语言</span><select value={selection.targetLanguage} onChange={(event) => setSelection({ ...selection, targetLanguage: event.target.value })}><option value="zh-Hans">简体中文</option><option value="en">English</option><option value="ja">日本語</option><option value="ko">한국어</option><option value="fr">Français</option></select></label>
    </div>
    <div className="pipeline-status"><span className={cloudOCR?.available ? "ready" : ""}>Vision {cloudOCR?.available ? "已就绪" : "未配置"}</span><span className={cloudTranslation?.available ? "ready" : ""}>Translation {cloudTranslation?.available ? "已就绪" : "未配置"}</span><span className={migan?.available ? "ready" : ""}>MI-GAN {migan?.available ? "已配置" : "未配置"}</span></div>
    {waitingForConfiguration ? <div className="pipeline-note">Google 凭证只放在 FastAPI 进程里；先配置 ADC 或 `GOOGLE_OCR_ACCESS_TOKEN`。</div> : null}
    {waitingForMIGAN ? <div className="pipeline-note">MI-GAN 只接受显式本地 `.mlpackage` 路径，不自动下载开发权重。</div> : null}
    <button className="run-button pipeline-run" onClick={onRun} disabled={busy || !canRun || waitingForConfiguration || waitingForMIGAN}>
      {busy ? <LoaderCircle size={17} className="spin" /> : <Play size={17} />} {busy ? "完整流程运行中" : "运行整套流程"}
    </button>
    {pipelineResult ? <div className="pipeline-complete"><Check size={14} /><span>最终译图已原子返回 · {formatMs(pipelineResult.timings_ms?.total)}</span><a href={`data:${pipelineResult.final_image_media_type};base64,${pipelineResult.final_image_base64}`} download={`${pipelineResult.image_name.replace(/\.[^.]+$/, "")}-translated.png`}>下载</a></div> : null}
  </div>;
}

function LanguagesIcon() {
  return <span className="language-icon">文</span>;
}

function ImageStage({ label, imageURL, result, showOverlay, zoom, pan, selectedRegion, onSelect }) {
  return <div className="stage-cell">
    {label ? <div className="stage-label">{label}{result ? ` · ${result.regions.length} 行 / ${result.groups?.length || 0} 块` : " · 等待运行"}</div> : null}
    <div className="image-stage" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom / 0.58})` }}>
      <img src={imageURL} alt={label ? `${label} 分析图` : "待分析图片"} />
      {showOverlay && result ? (
        <svg className="region-overlay" viewBox={`0 0 ${result.width} ${result.height}`} preserveAspectRatio="none">
          {result.regions.map((region, index) => {
            const color = REGION_COLORS[((region.group_id || index + 1) - 1) % REGION_COLORS.length];
            const selected = region.id === selectedRegion?.id;
            return (
              <g key={region.id} className={selected ? "selected-region" : ""} onClick={() => onSelect(region.id)}>
                <polygon points={region.quad.map((point) => `${point.x},${point.y}`).join(" ")} fill={`${color}${selected ? "22" : "0b"}`} stroke={color} strokeWidth={selected ? 4 : 2.4} vectorEffect="non-scaling-stroke" />
                {region.quad.map((point, pointIndex) => <circle key={pointIndex} cx={point.x} cy={point.y} r={selected ? 9 : 6} fill="white" stroke={color} strokeWidth="3" vectorEffect="non-scaling-stroke" />)}
                <g transform={`translate(${region.quad[0].x}, ${region.quad[0].y - 18})`}>
                  <rect x="-12" y="-15" width="29" height="25" rx="4" fill="white" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
                  <text x="2" y="3" fill={color} textAnchor="middle" fontSize="13" fontWeight="700">{region.group_id ? `${region.group_id}.${region.line_index}` : region.id}</text>
                </g>
              </g>
            );
          })}
        </svg>
      ) : null}
    </div>
  </div>;
}

function TimingRail({ timings, pipelineTimings }) {
  const stages = pipelineTimings ? [
    ["预处理", pipelineTimings.preprocessing],
    ["云 OCR", pipelineTimings.ocr],
    ["翻译", pipelineTimings.translation],
    ["修复", pipelineTimings.reconstruction],
    ["总计", pipelineTimings.total],
  ] : [
    ["预处理", timings?.preprocessing],
    ["检测", timings?.detection],
    ["方向", timings?.layout],
    ["识别", timings?.recognition],
    ["总计", timings?.total],
  ];
  const runtime = pipelineTimings ? "Google Cloud + 本地 MI-GAN · 最终图片原子返回 · 图片仅保留在当前会话" : "设备：本地 Python · 线程由 ONNX Runtime 管理 · 图片仅保留在当前会话";
  return <div className="timing-rail"><h3>阶段耗时（单张）</h3><div className="rail">{stages.map(([label, value], index) => <div className={`rail-stage ${index === stages.length - 1 ? "total" : ""}`} key={label}><span>{label}</span><i /><strong>{formatMs(value)}</strong></div>)}</div><div className="runtime-note">{runtime}</div></div>;
}

function ResultsTable({ result, selectedRegionID, onSelect }) {
  return <div className="results"><div className="results-heading"><h3>识别结果（{result?.regions.length || 0} 行 / {result?.groups?.length || 0} 翻译块）</h3><a className={result ? "" : "disabled"} href={result ? jsonDataURL(result) : undefined} download="ocr-lab-result.json"><Download size={15} /> JSON</a></div><div className="table-wrap"><table><thead><tr><th>ID</th><th>块 / 行</th><th>识别文本（原文）</th><th>脚本</th><th>方向</th><th>检测 / 识别</th></tr></thead><tbody>{result?.regions.map((region) => <tr key={region.id} className={selectedRegionID === region.id ? "selected" : ""} onClick={() => onSelect(region.id)}><td>{region.id}</td><td>{region.group_id ? `G${region.group_id} / L${region.line_index}` : "未连接"}</td><td>{region.text || "未识别"}</td><td>{scriptName(region.layout.script_family)}</td><td>{layoutName(region.layout)}</td><td>{region.confidence.toFixed(3)} / {region.recognition_confidence.toFixed(3)}</td></tr>)}</tbody></table></div></div>;
}

function jsonDataURL(result) {
  return `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(result, null, 2))}`;
}
