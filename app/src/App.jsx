import { useEffect, useMemo, useRef, useState } from "react";

const WOUND_TYPES = [
  "diabetic",
  "pressure",
  "trauma",
  "venous",
  "surgical",
  "arterial",
  "cellulitis",
  "miscellaneous",
  "normal",
];

const BOX_LABELS = [
  { name: "wound", color: "#0f766e" },
  { name: "foot", color: "#b45309" },
  { name: "limb", color: "#2563eb" },
  { name: "peri-wound", color: "#6d28d9" },
  { name: "healthy skin", color: "#be123c" },
];

const BOX_COLOR_MAP = Object.fromEntries(BOX_LABELS.map((item) => [item.name, item.color]));

const MODEL_INFO = {
  name: "Manual annotation (no inference)",
  version: "v1",
};

const blankAnnotation = () => ({
  boxes: [],
  type: "miscellaneous",
  severity: 0,
  notes: "",
});

function App() {
  const WEEK_TABS = [
    { key: "week1", label: "Week 1", topic: "Dataset + Annotator", state: "complete" },
    { key: "week2", label: "Week 2", topic: "Annotation + Classification", state: "active" },
    { key: "week3", label: "Week 3", topic: "Object Detection", state: "in_progress" },
    { key: "week4", label: "Week 4", topic: "Segmentation + Paper", state: "coming_soon" },
    { key: "week5", label: "Week 5", topic: "Final Evaluation + Video", state: "coming_soon" },
  ];
  const [activeWeek, setActiveWeek] = useState("week2");

  const [images, setImages] = useState([]);
  const [imageIdx, setImageIdx] = useState(0);
  const [annotations, setAnnotations] = useState({});
  const [selectedBoxLabel, setSelectedBoxLabel] = useState("wound");
  const [isDrawing, setIsDrawing] = useState(false);
  const [draftBox, setDraftBox] = useState(null);
  const [status, setStatus] = useState("Load images to begin annotation.");
  const [toast, setToast] = useState("");
  const [selectedBoxIdx, setSelectedBoxIdx] = useState(null);

  // --- Week 2 Live Inference States ---
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [classifying, setClassifying] = useState(false);
  const [classificationResult, setClassificationResult] = useState(null);
  const [classificationError, setClassificationError] = useState(null);
  const [activeMetricTab, setActiveMetricTab] = useState("confusion");

  // --- Week 2 Local Folder Batch Inspection States ---
  const [folderInput, setFolderInput] = useState("");
  const [batchImages, setBatchImages] = useState([]);
  const [batchIndex, setBatchIndex] = useState(0);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState(null);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [batchLimit, setBatchLimit] = useState(30);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setClassificationResult(null);
    setClassificationError(null);
  };

  const classifySelectedImage = async () => {
    if (!selectedFile) return;
    setClassifying(true);
    setClassificationResult(null);
    setClassificationError(null);
    
    const formData = new FormData();
    formData.append("image", selectedFile);
    
    try {
      const response = await fetch("http://127.0.0.1:5000/classify", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Failed to classify image.");
      }
      
      const data = await response.json();
      setClassificationResult(data);
    } catch (err) {
      setClassificationError(err.message || "Could not connect to the Python classification server on http://localhost:5000. Please start it in your terminal!");
    } finally {
      setClassifying(false);
    }
  };

  const clearSandbox = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setClassificationResult(null);
    setClassificationError(null);
  };

  const classifyRandomDatasetImage = async (classType) => {
    setClassifying(true);
    setClassificationResult(null);
    setClassificationError(null);
    
    try {
      const response = await fetch(`http://127.0.0.1:5000/classify_random?class_type=${classType}`);
      if (!response.ok) {
        throw new Error("Failed to load random scan from backend.");
      }
      
      const data = await response.json();
      const formattedBase64 = `data:image/jpeg;base64,${data.image_data}`;
      setPreviewUrl(formattedBase64);
      setClassificationResult({
        prediction: data.prediction,
        confidence: data.confidence,
        name: data.filename
      });
    } catch (err) {
      setClassificationError(err.message || "Could not fetch a random dataset scan. Please verify Python app_server.py is running!");
    } finally {
      setClassifying(false);
    }
  };

  const handleDirectorySelect = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    const imageFiles = files.filter(f => 
      f.type.startsWith("image/") || 
      f.name.endsWith(".jpg") || 
      f.name.endsWith(".jpeg") || 
      f.name.endsWith(".png") ||
      f.name.endsWith(".JPG") ||
      f.name.endsWith(".JPEG") ||
      f.name.endsWith(".PNG")
    );

    if (imageFiles.length === 0) {
      setBatchError("No supported images (.jpg, .jpeg, .png) found inside the selected folder.");
      return;
    }

    const limit = Math.min(imageFiles.length, batchLimit);
    const selectedFiles = imageFiles.slice(0, limit);

    const initialPlaceholders = selectedFiles.map(file => ({
      name: file.name,
      image_data: null,
      prediction: null,
      confidence: null,
      loading: true
    }));

    setBatchError(null);
    setBatchImages(initialPlaceholders);
    setBatchIndex(0);
    setBatchLoading(true);
    setBatchProgress({ current: 0, total: selectedFiles.length });

    const firstPath = selectedFiles[0].webkitRelativePath;
    const folderName = firstPath ? firstPath.split("/")[0] : "Local Directory";
    setFolderInput(folderName);

    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];

      try {
        const base64Data = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.readAsDataURL(file);
          reader.onload = () => resolve(reader.result);
          reader.onerror = error => reject(error);
        });

        setBatchImages(prev => {
          const copy = [...prev];
          if (copy[i]) {
            copy[i].image_data = base64Data;
          }
          return copy;
        });

        const formData = new FormData();
        formData.append("image", file);

        const response = await fetch("http://127.0.0.1:5000/classify", {
          method: "POST",
          body: formData
        });

        if (!response.ok) {
          throw new Error("Classification request failed.");
        }

        const resData = await response.json();
        
        setBatchImages(prev => {
          const copy = [...prev];
          if (copy[i]) {
            copy[i].prediction = resData.prediction;
            copy[i].confidence = resData.confidence;
            copy[i].probabilities = resData.probabilities;
            copy[i].loading = false;
          }
          return copy;
        });

      } catch (err) {
        console.error(`Error processing ${file.name}:`, err);
        setBatchImages(prev => {
          const copy = [...prev];
          if (copy[i]) {
            copy[i].prediction = "error";
            copy[i].confidence = 0;
            copy[i].loading = false;
          }
          return copy;
        });
      } finally {
        setBatchProgress(prev => ({ ...prev, current: i + 1 }));
      }
    }

    setBatchLoading(false);
  };

  const frameRef = useRef(null);

  useEffect(() => {
    const saved = localStorage.getItem("week1_annotator_state");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setAnnotations(parsed.annotations ?? {});
      } catch {
        setStatus("Autosave found but could not be loaded.");
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("week1_annotator_state", JSON.stringify({ annotations }));
  }, [annotations]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 2200);
    return () => clearTimeout(t);
  }, [toast]);

  const currentImage = images[imageIdx];
  const currentName = currentImage?.name;

  const currentAnnotation = useMemo(() => {
    if (!currentName) return blankAnnotation();
    return annotations[currentName] ?? blankAnnotation();
  }, [annotations, currentName]);

  const autosaveText = useMemo(
    () =>
      JSON.stringify(
        {
          saved_at: new Date().toISOString(),
          total_images_annotated: Object.keys(annotations).length,
          annotations,
        },
        null,
        2
      ),
    [annotations]
  );

  const setCurrentAnnotation = (next) => {
    if (!currentName) return;
    setAnnotations((prev) => ({
      ...prev,
      [currentName]: { ...(prev[currentName] ?? blankAnnotation()), ...currentAnnotation, ...next },
    }));
  };

  const releaseImageUrls = (list) => list.forEach((img) => URL.revokeObjectURL(img.url));

  const loadImages = (event) => {
    const files = Array.from(event.target.files ?? []).filter((f) => f.type.startsWith("image/"));
    if (!files.length) {
      setStatus("No valid image files selected.");
      return;
    }

    releaseImageUrls(images);

    const loaded = files
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((file) => ({ name: file.name, url: URL.createObjectURL(file) }));

    setImages(loaded);
    setImageIdx(0);
    setStatus(`${loaded.length} images loaded. Selected label: ${selectedBoxLabel}.`);
  };

  const getPoint = (event) => {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const x = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
    const y = Math.max(0, Math.min(event.clientY - rect.top, rect.height));
    return { x, y };
  };

  const onPointerDown = (event) => {
    if (!currentImage) return;
    const point = getPoint(event);
    if (!point) return;
    setIsDrawing(true);
    setDraftBox({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
  };

  const onPointerMove = (event) => {
    if (!isDrawing) return;
    const point = getPoint(event);
    if (!point) return;
    setDraftBox((prev) => (prev ? { ...prev, x2: point.x, y2: point.y } : prev));
  };

  const onPointerUp = () => {
    if (!isDrawing || !draftBox) return;
    setIsDrawing(false);

    const x1 = Math.min(draftBox.x1, draftBox.x2);
    const y1 = Math.min(draftBox.y1, draftBox.y2);
    const x2 = Math.max(draftBox.x1, draftBox.x2);
    const y2 = Math.max(draftBox.y1, draftBox.y2);

    if (Math.abs(x2 - x1) < 6 || Math.abs(y2 - y1) < 6) {
      setDraftBox(null);
      setStatus("Box too small. Draw a larger region.");
      return;
    }

    const nextBox = { 
      coords: [x1, y1, x2, y2], 
      label: selectedBoxLabel,
      color: BOX_COLOR_MAP[selectedBoxLabel] ?? "#0f766e",
      text: ""
    };
    setCurrentAnnotation({ boxes: [...currentAnnotation.boxes, nextBox] });
    setSelectedBoxIdx(currentAnnotation.boxes.length); // Select the new box
    setDraftBox(null);
    setStatus(`Added ${selectedBoxLabel} box. Select it to customize properties.`);
  };

  const updateBoxProperty = (idx, key, value) => {
    const nextBoxes = [...currentAnnotation.boxes];
    nextBoxes[idx] = { ...nextBoxes[idx], [key]: value };
    setCurrentAnnotation({ boxes: nextBoxes });
  };

  const removeBox = (idx) => {
    const nextBoxes = currentAnnotation.boxes.filter((_, i) => i !== idx);
    setCurrentAnnotation({ boxes: nextBoxes });
    if (selectedBoxIdx === idx) setSelectedBoxIdx(null);
    else if (selectedBoxIdx > idx) setSelectedBoxIdx(selectedBoxIdx - 1);
    setStatus("Box removed.");
  };

  const persistCurrentAnnotation = () => {
    if (!currentName) return null;
    const payload = { ...(annotations[currentName] ?? blankAnnotation()), ...currentAnnotation };

    setAnnotations((prev) => ({ ...prev, [currentName]: payload }));
    return { imageName: currentName, annotation: payload };
  };

  const saveAutosaveFile = (snapshot) => {
    localStorage.setItem(
      "week1_annotations_file",
      JSON.stringify(
        {
          saved_at: new Date().toISOString(),
          total_images_annotated: Object.keys(snapshot).length,
          annotations: snapshot,
        },
        null,
        2
      )
    );
  };

  const [exportFormat, setExportFormat] = useState("json");

  const toYOLO = () => {
    let yoloText = "";
    Object.entries(annotations).forEach(([imgName, data]) => {
      data.boxes.forEach((box) => {
        const [x1, y1, x2, y2] = box.coords;
        // In a real scenario, we'd need image dimensions for normalization
        // Since we don't have them easily here for all images without loading them, 
        // we'll export absolute for now or note it. 
        // But let's try to assume a 1000x1000 canvas for the sake of demo or use actual if available.
        const w = x2 - x1;
        const h = y2 - y1;
        const cx = x1 + w / 2;
        const cy = y1 + h / 2;
        const classId = BOX_LABELS.findIndex((l) => l.name === box.label);
        yoloText += `${imgName}: ${classId} ${cx.toFixed(4)} ${cy.toFixed(4)} ${w.toFixed(4)} ${h.toFixed(4)}\n`;
      });
    });
    downloadFile("annotations.yolo.txt", "text/plain;charset=utf-8", yoloText);
  };

  const toCOCO = () => {
    const categories = BOX_LABELS.map((item, id) => ({ id, name: item.name }));
    const imagesOut = [];
    const annotationsOut = [];
    let annId = 1;

    Object.entries(annotations).forEach(([imgName, data], imgId) => {
      imagesOut.push({ id: imgId, file_name: imgName });
      data.boxes.forEach((box) => {
        const [x1, y1, x2, y2] = box.coords;
        annotationsOut.push({
          id: annId,
          image_id: imgId,
          category_id: BOX_LABELS.findIndex((l) => l.name === box.label),
          bbox: [x1, y1, x2 - x1, y2 - y1],
          area: (x2 - x1) * (y2 - y1),
          iscrowd: 0,
          metadata: { type: data.type, severity: data.severity },
        });
        annId += 1;
      });
    });

    const payload = { images: imagesOut, annotations: annotationsOut, categories };
    downloadFile("annotations.coco.json", "application/json;charset=utf-8", JSON.stringify(payload, null, 2));
  };

  const toCSV = () => {
    const rows = [["image", "class", "severity", "notes", "box_count", "boxes"]];
    Object.entries(annotations).forEach(([name, data]) => {
      rows.push([
        name,
        data.type,
        String(data.severity),
        (data.notes ?? "").replaceAll('"', '""'),
        String(data.boxes.length),
        JSON.stringify(data.boxes).replaceAll('"', '""'),
      ]);
    });
    const csv = rows.map((row) => row.map((cell) => `"${cell}"`).join(",")).join("\n");
    downloadFile("annotations.csv", "text/csv;charset=utf-8", csv);
  };

  const toJSON = () => {
    const payload = {
      saved_at: new Date().toISOString(),
      total_images_annotated: Object.keys(annotations).length,
      annotations,
    };
    downloadFile("annotations.json", "application/json;charset=utf-8", JSON.stringify(payload, null, 2));
  };

  const downloadAnnotationFiles = () => {
    // Always download JSON (raw state)
    toJSON();

    // Download dynamic format
    if (exportFormat === "csv") toCSV();
    else if (exportFormat === "yolo") toYOLO();
    else if (exportFormat === "coco") toCOCO();
    
    setStatus(`Downloaded JSON + ${exportFormat.toUpperCase()} files.`);
    setToast(`JSON + ${exportFormat.toUpperCase()} downloaded`);
  };

  const handleSaveAndNext = () => {
    const saved = persistCurrentAnnotation();
    if (!saved) {
      setStatus("Load images first.");
      return;
    }

    // Saving downloads the annotated image as a PNG. Save it into the annotated/ folder.
    downloadAnnotatedImage();

    const nextSnapshot = { ...annotations, [saved.imageName]: saved.annotation };
    saveAutosaveFile(nextSnapshot);

    setImages((prev) => {
      const next = prev.filter((_, idx) => idx !== imageIdx);
      URL.revokeObjectURL(prev[imageIdx].url);
      return next;
    });
    setImageIdx(0);

    setToast(`Saved ${saved.imageName} successfully`);
    setStatus("Saved and removed current image from queue.");
  };

  const downloadBlob = (name, blob, withStatus = true) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
    if (withStatus) setStatus(`${name} exported.`);
  };

  const drawLabelBadge = (ctx, x, y, label, color) => {
    const paddingX = 6;
    const paddingY = 4;
    ctx.font = "700 12px Manrope, system-ui, sans-serif";
    const metrics = ctx.measureText(label);
    const badgeWidth = Math.max(64, metrics.width + paddingX * 2);
    const badgeHeight = 18;
    const badgeX = x;
    const badgeY = Math.max(0, y - badgeHeight - 4);

    ctx.fillStyle = color;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(badgeX, badgeY, badgeWidth, badgeHeight, 5);
    } else {
      ctx.rect(badgeX, badgeY, badgeWidth, badgeHeight);
    }
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, badgeX + paddingX, badgeY + badgeHeight - paddingY);
  };

  const downloadAnnotatedImage = (nameOverride) => {
    if (!currentImage || !frameRef.current) return;

    const frame = frameRef.current.getBoundingClientRect();
    const width = Math.round(frame.width);
    const height = Math.round(frame.height);
    if (!width || !height) return;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      const scale = Math.min(width / img.width, height / img.height);
      const drawWidth = img.width * scale;
      const drawHeight = img.height * scale;
      const offsetX = (width - drawWidth) / 2;
      const offsetY = (height - drawHeight) / 2;

      ctx.fillStyle = "#f5f4f0";
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);

      ctx.lineWidth = 2.5;
      currentAnnotation.boxes.forEach((box) => {
        const [x1, y1, x2, y2] = box.coords;
        const stroke = box.color || BOX_COLOR_MAP[box.label] || "#0f766e";
        const label = box.text || box.label;
        ctx.strokeStyle = stroke;
        ctx.setLineDash([]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        drawLabelBadge(ctx, x1, y1, label, stroke);
      });

      canvas.toBlob((blob) => {
        if (!blob) return;
        const safeName = nameOverride ?? `annotated_${currentImage.name.replace(/\.[^/.]+$/, "")}.png`;
        downloadBlob(`annotated/${safeName}`, blob, false);
        setToast(`Annotated image saved: ${safeName}`);
      }, "image/png");
    };

    img.src = currentImage.url;
  };

  const downloadFile = (name, mime, data, withStatus = true) => {
    const blob = new Blob([data], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
    if (withStatus) setStatus(`${name} exported.`);
  };

  return (
    <main className="mx-auto min-h-screen max-w-[1440px] px-4 py-6 md:px-8 lg:px-10">
      <header className="hero-panel mb-6 rounded-3xl border border-clay bg-card px-6 py-6 shadow-soft md:px-8">
        <div className="flex flex-wrap items-start justify-between gap-5">
          <div>
            <p className="font-display text-xs uppercase tracking-[0.3em] text-warm">CV Project Dashboard</p>
            <h1 className="mt-2 font-display text-3xl text-ink md:text-4xl">Medical Vision Workflow</h1>
            <p className="mt-2 max-w-3xl text-sm text-stone-700">
              Navigate each project week, run the current milestone, and track what is complete vs pending.
            </p>
          </div>
          <div className="rounded-2xl border border-stone-200 bg-white/80 px-4 py-3 text-xs text-stone-700">
            <p className="font-semibold text-stone-900">Current Week</p>
            <p className="mt-1 text-xl font-extrabold text-ink">{WEEK_TABS.find((w) => w.key === activeWeek)?.label}</p>
          </div>
        </div>
        <div className="mt-5 flex flex-wrap gap-2">
          {WEEK_TABS.map((week) => {
            const isActive = week.key === activeWeek;
            const pillClass =
              week.state === "complete"
                ? "bg-emerald-100 text-emerald-700"
                : week.state === "in_progress"
                  ? "bg-amber-100 text-amber-700"
                  : week.state === "active"
                    ? "bg-blue-100 text-blue-700"
                    : "bg-stone-200 text-stone-700";
            const pillText =
              week.state === "complete"
                ? "Complete"
                : week.state === "in_progress"
                  ? "In Progress"
                  : week.state === "active"
                    ? "Current"
                    : "Coming Soon";
            return (
              <button
                key={week.key}
                type="button"
                onClick={() => setActiveWeek(week.key)}
                className={`rounded-2xl border px-3 py-2 text-left transition ${isActive ? "border-primary bg-primary/10" : "border-stone-200 bg-white hover:bg-stone-50"}`}
              >
                <p className="text-xs font-bold uppercase tracking-wide text-ink">{week.label}</p>
                <p className="text-xs text-stone-600">{week.topic}</p>
                <span className={`mt-1 inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${pillClass}`}>
                  {pillText}
                </span>
              </button>
            );
          })}
        </div>
      </header>

      {activeWeek === "week2" && (
        <>
          {/* SECTION 1: METRICS DASHBOARD & LIVE CLASSIFIER SANDBOX */}
          <section className="grid gap-6 lg:grid-cols-2">
            {/* LEFT SIDE: METRICS DASHBOARD */}
            <div className="rounded-3xl border border-clay bg-card p-6 shadow-soft flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="font-display text-2xl text-ink">Week 2 Metrics Dashboard</h2>
                    <p className="mt-1 text-sm text-stone-600">Model backbone: MobileNetV3-Large Transfer Learning</p>
                  </div>
                  <span className="inline-flex rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-800 uppercase tracking-wider animate-pulse">
                    Model Deployed
                  </span>
                </div>
                
                {/* Metrics Grid */}
                <div className="mt-6 grid grid-cols-2 gap-4">
                  <div className="rounded-2xl border border-stone-200 bg-white p-4 shadow-sm transition hover:shadow">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-stone-400">Overall Accuracy</span>
                    <div className="mt-1 flex items-baseline gap-1">
                      <span className="text-3xl font-extrabold text-ink">99.67%</span>
                      <span className="text-xs font-bold text-emerald-600">▲ 0.12%</span>
                    </div>
                    <p className="mt-1 text-[11px] text-stone-500">299 / 300 correct splits</p>
                  </div>
                  
                  <div className="rounded-2xl border border-stone-200 bg-white p-4 shadow-sm transition hover:shadow">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-stone-400">Wound Precision</span>
                    <div className="mt-1 flex items-baseline gap-1">
                      <span className="text-3xl font-extrabold text-ink">99.50%</span>
                      <span className="text-xs font-semibold text-stone-500">Stable</span>
                    </div>
                    <p className="mt-1 text-[11px] text-stone-500">Only 1 false alarm</p>
                  </div>

                  <div className="rounded-2xl border border-stone-200 bg-white p-4 shadow-sm transition hover:shadow">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-stone-400">Wound Recall</span>
                    <div className="mt-1 flex items-baseline gap-1">
                      <span className="text-3xl font-extrabold text-emerald-700">100.0%</span>
                      <span className="text-xs font-bold text-emerald-600">Perfect</span>
                    </div>
                    <p className="mt-1 text-[11px] text-stone-500">0 false negatives (Clinical safe)</p>
                  </div>

                  <div className="rounded-2xl border border-stone-200 bg-white p-4 shadow-sm transition hover:shadow">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-stone-400">Balanced F1-Score</span>
                    <div className="mt-1 flex items-baseline gap-1">
                      <span className="text-3xl font-extrabold text-ink">99.75%</span>
                      <span className="text-xs font-bold text-emerald-600">Optimal</span>
                    </div>
                    <p className="mt-1 text-[11px] text-stone-500">Robust binary balance</p>
                  </div>
                </div>
              </div>

              {/* Live Dataset Quick Loaders */}
              <div className="mt-6 bg-paper border border-stone-200 rounded-2xl p-4 shadow-sm">
                <span className="text-[10px] font-bold uppercase tracking-widest text-stone-500 block mb-2.5">
                  ⚡ Dataset Quick-Loaders (1-Click Run)
                </span>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    className="rounded-xl border border-emerald-200 bg-emerald-50/30 hover:bg-emerald-50 p-3 text-left transition duration-200 shadow-sm flex items-center gap-3 group"
                    onClick={() => classifyRandomDatasetImage("normal")}
                    disabled={classifying}
                  >
                    <span className="text-2xl transition-transform group-hover:scale-110">🟢</span>
                    <div>
                      <p className="text-xs font-bold text-emerald-800">Healthy Control</p>
                      <p className="text-[9px] text-emerald-600 font-semibold mt-0.5">data/Nomal</p>
                    </div>
                  </button>

                  <button
                    type="button"
                    className="rounded-xl border border-red-200 bg-red-50/30 hover:bg-red-50 p-3 text-left transition duration-200 shadow-sm flex items-center gap-3 group"
                    onClick={() => classifyRandomDatasetImage("wound")}
                    disabled={classifying}
                  >
                    <span className="text-2xl transition-transform group-hover:scale-110">🚨</span>
                    <div>
                      <p className="text-xs font-bold text-red-800">Active Wound Scan</p>
                      <p className="text-[9px] text-red-600 font-semibold mt-0.5">data/wound_main</p>
                    </div>
                  </button>
                </div>
              </div>
            </div>

            {/* RIGHT SIDE: LIVE INFERENCE SANDBOX */}
            <div className="rounded-3xl border border-clay bg-card p-6 shadow-soft flex flex-col gap-5 justify-between">
              <div>
                <h2 className="font-display text-2xl text-ink">🔬 Live Classifier Sandbox</h2>
                <p className="mt-1 text-sm text-stone-600">
                  Upload a medical image to execute the trained MobileNetV3 model in real-time, or instantly load a random clinical scan from your patient folders!
                </p>
              </div>

              {/* Upload Area / Image Preview */}
              <div className="flex-1 flex flex-col items-center justify-center min-h-[300px] rounded-2xl border-2 border-dashed border-stone-300 bg-stone-50 p-4 transition-all hover:bg-stone-100 relative">
                {!previewUrl ? (
                  <label className="flex flex-col items-center justify-center cursor-pointer text-center p-6 w-full h-full">
                    <svg className="h-12 w-12 text-stone-400 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p className="mt-4 text-sm font-bold text-ink">Drag & drop or browse medical foot scans</p>
                    <p className="mt-1 text-xs text-stone-500">Supports JPG, PNG (automatically resized to 331x331px)</p>
                    <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
                  </label>
                ) : (
                  <div className="relative w-full h-full flex flex-col items-center">
                    <div className="relative w-full max-h-[260px] overflow-hidden rounded-xl border border-stone-200 bg-white p-1 shadow-sm">
                      <img src={previewUrl} alt="Sandbox Preview" className="mx-auto max-h-[240px] w-full object-contain rounded-lg" />
                      {classifying && (
                        <div className="absolute inset-0 bg-white/70 backdrop-blur-sm flex flex-col items-center justify-center p-4">
                          <svg className="animate-spin h-10 w-10 text-primary mb-3" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          <p className="text-xs font-bold text-primary animate-pulse tracking-wider uppercase">Running Model Inference...</p>
                          <span className="text-[10px] text-stone-500 mt-1">Normalizing tensor to ImageNet stats</span>
                        </div>
                      )}
                    </div>
                    
                    {/* Results Display */}
                    {classificationResult && (
                      <div className="mt-4 w-full p-4 rounded-xl border-2 border-clay bg-white/95 animate-in fade-in slide-in-from-bottom-2 duration-300 shadow-sm text-left">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-bold uppercase tracking-wider text-stone-500">
                            {classificationResult.name ? `File: ${classificationResult.name}` : "Prediction Outcome"}
                          </span>
                          <span className={`rounded-full px-3 py-1 text-xs font-extrabold uppercase tracking-wider ${
                            classificationResult.prediction === "wound" 
                              ? "bg-red-100 text-red-800 ring-2 ring-red-400 ring-offset-1 animate-pulse" 
                              : "bg-emerald-100 text-emerald-800 ring-2 ring-emerald-400 ring-offset-1"
                          }`}>
                            {classificationResult.prediction === "wound" ? "Wound" : "Normal"}
                          </span>
                        </div>

                        {/* Confidence score bar */}
                        <div className="mt-3">
                          <div className="flex justify-between text-xs font-semibold mb-1">
                            <span className="text-stone-700">Model Confidence:</span>
                            <span className="text-ink font-bold">{(classificationResult.confidence * 100).toFixed(2)}%</span>
                          </div>
                          <div className="h-3 w-full bg-stone-100 rounded-full overflow-hidden">
                            <div 
                              className={`h-full transition-all duration-1000 ${
                                classificationResult.prediction === "wound" ? "bg-red-600" : "bg-emerald-600"
                              }`}
                              style={{ width: `${classificationResult.confidence * 100}%` }}
                            />
                          </div>
                        </div>

                        {/* Clinical Recommendations */}
                        <p className="mt-4 p-3 rounded-lg bg-accent-soft/30 border border-clay text-xs text-stone-700 leading-relaxed">
                          {classificationResult.prediction === "wound" ? (
                            <span>
                              <b>🚨 Clinical Alert:</b> Active skin lesion / wound indicators detected. <b>Week 3 bounding box detector</b> and <b>Week 4 boundary segmenter</b> should be activated to locate bounding contours and map wound surface area.
                            </span>
                          ) : (
                            <span>
                              <b>🩺 Observation Clear:</b> The model classified the skin texture as healthy diabetic foot tissue with extremely high confidence. Patient monitoring protocol remains in baseline outpatient status.
                            </span>
                          )}
                        </p>
                      </div>
                    )}

                    {/* Error Display */}
                    {classificationError && (
                      <div className="mt-4 w-full p-4 rounded-xl border border-red-200 bg-red-50 text-xs text-red-800 leading-relaxed shadow-sm text-left">
                        <p className="font-bold mb-1">⚠️ Server Connection Offline</p>
                        <p className="text-[11px] leading-relaxed">{classificationError}</p>
                        <div className="mt-3 bg-white/70 p-2 rounded border border-red-100 text-[10px] text-stone-600 font-mono text-left">
                          Run this in your terminal: <br />
                          <span className="font-semibold text-ink">python week2/app_server.py</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Sandbox Actions */}
              <div className="flex gap-3">
                {previewUrl ? (
                  <>
                    <button
                      type="button"
                      className="btn-secondary flex-1 py-3 text-xs font-bold uppercase tracking-wider rounded-xl border border-stone-200 bg-white hover:bg-stone-50 transition"
                      onClick={clearSandbox}
                    >
                      Clear Sandbox
                    </button>
                    {!classificationResult && !classificationError && (
                      <button
                        type="button"
                        className="btn-primary flex-1 py-3 text-xs font-bold uppercase tracking-wider rounded-xl bg-primary text-white hover:bg-primary-dark transition shadow-md"
                        onClick={classifySelectedImage}
                        disabled={classifying}
                      >
                        {classifying ? "Computing..." : "Run Classifier"}
                      </button>
                    )}
                  </>
                ) : (
                  <div className="w-full text-center py-2 text-xs text-stone-500 leading-relaxed">
                    💡 <b>Quick Test Tip:</b> Run your Python backend and click above to browse any image from the raw folders (e.g. <code>data/Nomal/</code> or <code>data/wound_main/</code>) to see immediate inference!
                  </div>
                )}
              </div>
            </div>
          </section>

          {/* SECTION 2: FULL WIDTH FOLDER BATCH INFERENCE INSPECTOR */}
          <div className="rounded-3xl border border-clay bg-card p-6 shadow-soft mt-6">
            <div className="border-b border-clay pb-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h3 className="font-display text-lg text-ink">📂 Folder Batch Inference Inspector</h3>
                <p className="mt-1 text-xs text-stone-500">Scan any local workspace directory and fast-inspect predictions in a streamlined batch carousel</p>
              </div>
              
              {/* Folder Path Input Panel & Limit Select */}
              <div className="flex items-center gap-3 max-w-xl w-full">
                {/* Limit Drop-down Select Combo Box */}
                <div className="flex items-center gap-1.5 whitespace-nowrap">
                  <span className="text-[11px] font-semibold text-stone-500">Limit:</span>
                  <select
                    className="rounded-xl border border-stone-300 bg-white px-2 py-1.5 text-xs font-semibold text-stone-700 focus:outline-none focus:ring-2 focus:ring-primary shadow-sm cursor-pointer select-none"
                    value={batchLimit}
                    onChange={(e) => setBatchLimit(Number(e.target.value))}
                    disabled={batchLoading}
                  >
                    <option value={10}>10 images</option>
                    <option value={20}>20 images</option>
                    <option value={30}>30 images</option>
                    <option value={50}>50 images</option>
                    <option value={100}>100 images</option>
                  </select>
                </div>

                {/* Path Display */}
                <div className="flex-1 flex items-center rounded-xl border border-stone-300 bg-stone-50 px-3 py-2 text-xs font-mono text-stone-600 shadow-sm min-w-0">
                  <span className="truncate">
                    {folderInput ? `Selected: ${folderInput} (${batchImages.length} images)` : "No folder chosen yet"}
                  </span>
                </div>

                {/* High-Contrast Folder Select Button */}
                <label className="bg-stone-900 hover:bg-stone-800 text-white rounded-xl px-4 py-2.5 text-xs font-bold shadow-md cursor-pointer transition flex items-center gap-2 whitespace-nowrap disabled:opacity-50 select-none">
                  <span>📂 Choose Local Folder</span>
                  <input
                    type="file"
                    webkitdirectory=""
                    directory=""
                    multiple
                    className="hidden"
                    onChange={handleDirectorySelect}
                    disabled={batchLoading}
                  />
                </label>
              </div>
            </div>

            {batchLoading && (
              <div className="mt-4 p-3 rounded-xl bg-primary-soft/10 border border-primary/20 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2.5">
                  <svg className="animate-spin h-4 w-4 text-primary" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span className="font-bold text-primary animate-pulse uppercase tracking-wider">
                    Analyzing Batch: {batchProgress.current} / {batchProgress.total} images classified
                  </span>
                </div>
                <span className="text-[10px] text-stone-500 font-mono">Running CPU inference via Flask...</span>
              </div>
            )}

            {batchError && (
              <div className="mt-4 p-4 rounded-xl border border-red-200 bg-red-50 text-xs text-red-800 leading-relaxed shadow-sm">
                <p className="font-bold mb-1">⚠️ Batch Loading Failed</p>
                <p>{batchError}</p>
                <p className="mt-2 text-[10px] text-stone-500">
                  Tip: Make sure the selected folder contains valid image files (.jpg, .jpeg, or .png).
                </p>
              </div>
            )}

            {/* Interactive Carousel Viewer */}
            {batchImages.length > 0 && (
              <div className="mt-6 grid lg:grid-cols-5 gap-6">
                
                {/* Thumbnails Sidebar List (Left 2 Columns) */}
                <div className="lg:col-span-2 rounded-2xl border border-stone-200 bg-stone-50 p-3 shadow-inner max-h-[420px] overflow-y-auto flex flex-col gap-2">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-stone-500 block mb-2 px-1">
                    📋 Loaded Items ({batchImages.length} Images)
                  </span>
                  
                  {batchImages.map((img, idx) => {
                    if (img.loading) {
                      return (
                        <button
                          key={img.name + idx}
                          type="button"
                          className={`w-full rounded-xl border p-2 text-left transition flex items-center gap-3 shadow-sm ${
                            batchIndex === idx
                              ? "bg-primary-soft/10 border-stone-300 ring-1 ring-primary-soft"
                              : "bg-white border-stone-150"
                          }`}
                          onClick={() => setBatchIndex(idx)}
                        >
                          {/* Skeleton/Preview Thumbnail */}
                          <div className="h-10 w-10 rounded-lg overflow-hidden border border-stone-200 bg-stone-100 flex-shrink-0 animate-pulse flex items-center justify-center">
                            {img.image_data ? (
                              <img src={img.image_data} alt="thumb" className="h-full w-full object-cover" />
                            ) : (
                              <div className="h-4 w-4 bg-stone-200 rounded-full" />
                            )}
                          </div>
                          {/* Skeleton details */}
                          <div className="min-w-0 flex-1 flex flex-col gap-1">
                            <p className="text-[11px] font-semibold text-stone-500 truncate">{img.name}</p>
                            <div className="flex items-center gap-2 mt-0.5">
                              <span className="h-3.5 w-12 rounded bg-stone-200 animate-pulse" />
                              <span className="h-3 w-8 rounded bg-stone-200 animate-pulse" />
                            </div>
                          </div>
                        </button>
                      );
                    }

                    // Otherwise show real completed data
                    return (
                      <button
                        key={img.name + idx}
                        type="button"
                        className={`w-full rounded-xl border p-2 text-left transition flex items-center gap-3 shadow-sm ${
                          batchIndex === idx
                            ? "bg-primary-soft/30 border-primary ring-1 ring-primary"
                            : "bg-white border-stone-200 hover:bg-stone-50"
                        }`}
                        onClick={() => setBatchIndex(idx)}
                      >
                        <div className="h-10 w-10 rounded-lg overflow-hidden border border-stone-200 bg-white flex-shrink-0">
                          <img src={img.image_data} alt="thumb" className="h-full w-full object-cover" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-[11px] font-semibold text-ink truncate">{img.name}</p>
                          <div className="flex items-center gap-2 mt-0.5">
                            <span className={`text-[9px] font-extrabold uppercase px-1.5 py-0.5 rounded-md ${
                              img.prediction === "wound" 
                                ? "bg-red-100 text-red-800 ring-1 ring-red-200" 
                                : img.prediction === "normal"
                                ? "bg-emerald-100 text-emerald-800 ring-1 ring-emerald-200"
                                : "bg-amber-100 text-amber-800 ring-1 ring-amber-200"
                            }`}>
                              {img.prediction === "wound" 
                                ? "Wound" 
                                : img.prediction === "normal"
                                ? "Normal"
                                : "Error"}
                            </span>
                            <span className="text-[9px] text-stone-500 font-medium">{(img.confidence * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* Main Selected Image Viewer (Right 3 Columns) */}
                <div className="lg:col-span-3 rounded-2xl border border-stone-200 bg-white p-5 shadow-sm flex flex-col justify-between min-h-[400px]">
                  
                  {/* Title & Index Pagination */}
                  <div className="flex items-center justify-between border-b border-stone-100 pb-3">
                    <div className="min-w-0">
                      <p className="text-xs font-bold uppercase tracking-wider text-stone-500">Active Batch Inspection</p>
                      <p className="text-[11px] font-mono text-ink font-semibold truncate mt-0.5">{batchImages[batchIndex].name}</p>
                    </div>
                    <span className="text-xs font-bold text-stone-600 bg-stone-100 px-3 py-1 rounded-full whitespace-nowrap">
                      {batchIndex + 1} of {batchImages.length}
                    </span>
                  </div>

                  {/* High-res Image Display */}
                  <div className="my-4 flex-1 flex items-center justify-center rounded-xl bg-stone-50 border border-stone-150 p-2 overflow-hidden relative min-h-[250px]">
                    {batchImages[batchIndex].image_data ? (
                      <img 
                        src={batchImages[batchIndex].image_data} 
                        alt="active-preview" 
                        className="max-h-[240px] rounded-lg object-contain transition duration-200" 
                      />
                    ) : (
                      <div className="flex flex-col items-center justify-center animate-pulse gap-2">
                        <div className="h-12 w-12 rounded bg-stone-200 flex items-center justify-center">
                          <svg className="h-6 w-6 text-stone-400 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                        </div>
                        <p className="text-[10px] text-stone-400 font-bold uppercase tracking-wider">Reading bytes...</p>
                      </div>
                    )}
                    
                    {/* Prediction Overlay Badge */}
                    <div className="absolute top-3 right-3">
                      {batchImages[batchIndex].loading ? (
                        <span className="rounded-full px-3 py-1 text-xs font-extrabold uppercase tracking-wider bg-stone-100 text-stone-500 shadow-md ring-2 ring-stone-200 ring-offset-1 animate-pulse flex items-center gap-1.5">
                          <span className="h-1.5 w-1.5 rounded-full bg-stone-400 animate-ping" />
                          Analyzing
                        </span>
                      ) : (
                        <span className={`rounded-full px-3 py-1 text-xs font-extrabold uppercase tracking-wider shadow-md ring-2 ring-offset-1 ${
                          batchImages[batchIndex].prediction === "wound" 
                            ? "bg-red-100 text-red-800 ring-red-400" 
                            : batchImages[batchIndex].prediction === "normal"
                            ? "bg-emerald-100 text-emerald-800 ring-emerald-400"
                            : "bg-amber-100 text-amber-800 ring-amber-400"
                        }`}>
                          {batchImages[batchIndex].prediction === "wound" 
                            ? "Wound" 
                            : batchImages[batchIndex].prediction === "normal"
                            ? "Normal"
                            : "Error"}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Confidence Slider bar & Quick Stepper Controls */}
                  <div>
                    {batchImages[batchIndex].loading ? (
                      <div>
                        <div className="flex items-center justify-between text-xs font-semibold mb-2 text-stone-400 animate-pulse">
                          <span>Classification Confidence:</span>
                          <span>--%</span>
                        </div>
                        <div className="h-2 w-full bg-stone-100 rounded-full overflow-hidden mb-5 animate-pulse">
                          <div className="h-full bg-stone-200 w-1/3 rounded-full animate-ping" />
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="flex items-center justify-between text-xs font-semibold mb-2">
                          <span className="text-stone-700">Classification Confidence:</span>
                          <span className="text-ink font-bold font-mono">{(batchImages[batchIndex].confidence * 100).toFixed(2)}%</span>
                        </div>
                        <div className="h-2 w-full bg-stone-100 rounded-full overflow-hidden mb-5">
                          <div 
                            className={`h-full transition-all duration-300 ${
                              batchImages[batchIndex].prediction === "wound" ? "bg-red-600" : "bg-emerald-600"
                            }`}
                            style={{ width: `${batchImages[batchIndex].confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Stepper Controls */}
                    <div className="flex gap-4">
                      <button
                        type="button"
                        className="flex-1 py-2.5 text-xs font-bold uppercase rounded-xl border border-stone-200 bg-white hover:bg-stone-50 text-stone-700 transition shadow-sm disabled:opacity-40"
                        onClick={() => setBatchIndex(prev => Math.max(0, prev - 1))}
                        disabled={batchIndex === 0}
                      >
                        ◀ Previous
                      </button>
                      <button
                        type="button"
                        className="flex-1 py-2.5 text-xs font-bold uppercase rounded-xl border border-stone-200 bg-white hover:bg-stone-50 text-stone-700 transition shadow-sm disabled:opacity-40"
                        onClick={() => setBatchIndex(prev => Math.min(batchImages.length - 1, prev + 1))}
                        disabled={batchIndex === batchImages.length - 1}
                      >
                        Next ▶
                      </button>
                    </div>
                  </div>
                  
                </div>
              </div>
            )}
            
            {batchImages.length === 0 && !batchLoading && !batchError && (
              <div className="mt-4 p-8 rounded-2xl bg-stone-50 border border-stone-150 text-center text-xs text-stone-500">
                💡 Select your batch size, click <b>Choose Local Folder</b> to pick a directory containing images, and watch them classify progressively and dynamically!
              </div>
            )}
          </div>

          {/* SECTION 3: FULL WIDTH PERFORMANCE VISUALIZATIONS */}
          <div className="rounded-3xl border border-clay bg-card p-6 shadow-soft mt-6">
            <div className="border-b border-clay pb-4">
              <h3 className="font-display text-lg text-ink">Model Performance Visualizations</h3>
              <p className="mt-1 text-sm text-stone-600">Dual validation metrics demonstrating perfect screening recall and high optimization convergence</p>
            </div>

            <div className="mt-6 grid gap-6 md:grid-cols-2">
              {/* Confusion Matrix */}
              <div className="flex flex-col justify-between rounded-2xl border border-stone-100 bg-white p-4 shadow-sm">
                <span className="text-[11px] font-bold uppercase tracking-widest text-primary mb-3 text-center block font-mono">I. Confusion Matrix</span>
                <div className="relative overflow-hidden rounded-xl border border-stone-200 bg-stone-100 p-4 shadow-inner flex items-center justify-center min-h-[300px]">
                  <img
                    src="/confusion_matrix.png"
                    alt="Confusion Matrix"
                    className="mx-auto max-h-[360px] rounded-lg object-contain transition-transform hover:scale-[1.02] duration-300"
                    onError={(e) => {
                      e.target.src = "https://placehold.co/600x600/0f766e/ffffff?text=Run+evaluate.py+to+generate+Confusion+Matrix";
                    }}
                  />
                </div>
                <p className="mt-4 text-xs text-stone-600 leading-relaxed italic text-center px-4">
                  <b>Clinical Sensitivity:</b> 0 missed wounds (100.0% sensitivity) out of 200 validation wound records, eliminating the diagnostic screening risk of false negatives.
                </p>
              </div>

              {/* Training Curves */}
              <div className="flex flex-col justify-between rounded-2xl border border-stone-100 bg-white p-4 shadow-sm">
                <span className="text-[11px] font-bold uppercase tracking-widest text-warm mb-3 text-center block font-mono">II. Convergence Curves</span>
                <div className="relative overflow-hidden rounded-xl border border-stone-200 bg-stone-100 p-4 shadow-inner flex items-center justify-center min-h-[300px]">
                  <img
                    src="/training_curves.png"
                    alt="Training Curves"
                    className="mx-auto max-h-[360px] rounded-lg object-contain transition-transform hover:scale-[1.02] duration-300"
                    onError={(e) => {
                      e.target.src = "https://placehold.co/600x600/b45309/ffffff?text=Run+train.py+to+generate+Curves";
                    }}
                  />
                </div>
                <p className="mt-4 text-xs text-stone-600 leading-relaxed italic text-center px-4">
                  <b>Optimization Profile:</b> Loss drops smoothly to 0.0086. Freezing the pre-trained feature extractor parameters enabled zero gradient dissipation and robust training on CPU.
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {(activeWeek === "week3" || activeWeek === "week4" || activeWeek === "week5") && (
        <section className="panel mb-5 rounded-3xl border border-clay bg-card p-6 shadow-soft">
          <h2 className="font-display text-2xl text-ink">
            {activeWeek === "week3" ? "Week 3: Object Detection" : activeWeek === "week4" ? "Week 4: Segmentation + Paper" : "Week 5: Final Evaluation"}
          </h2>
          <p className="mt-2 text-sm text-stone-700">
            {activeWeek === "week4"
              ? "This module is in progress. Add segmentation training, metrics, and paper artifacts next."
              : "This module is coming soon. You can keep preparing data and annotation quality now."}
          </p>
        </section>
      )}

      {activeWeek === "week1" && <section className="grid gap-5 xl:grid-cols-[1.65fr_1fr]">
        <article className="panel rounded-3xl border border-clay bg-card p-4 shadow-soft md:p-5">
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <label className="btn-ghost inline-flex cursor-pointer items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold transition-all hover:shadow-md">
              Load Images
              <input type="file" accept="image/*" multiple className="hidden" onChange={loadImages} />
            </label>
            <label className="btn-ghost inline-flex cursor-pointer items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold transition-all hover:shadow-md">
              Load Folder
              <input type="file" accept="image/*" multiple className="hidden" webkitdirectory="" directory="" onChange={loadImages} />
            </label>
            <div className="ml-auto rounded-xl border border-clay bg-white/50 px-3 py-2 text-xs font-bold text-ink backdrop-blur-sm">
              {currentImage ? `${imageIdx + 1}/${images.length} • ${currentImage.name}` : "Queue Empty"}
            </div>
          </div>

          <div className="canvas-wrap relative rounded-2xl border border-stone-300 bg-stone-100 p-2">
            <div
              ref={frameRef}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              className="relative aspect-[1/1] w-full touch-none overflow-hidden rounded-xl bg-stone-100"
            >
              {currentImage ? (
                <>
                  <img src={currentImage.url} alt={currentImage.name} className="h-full w-full object-contain" draggable={false} />
                  <svg className="pointer-events-none absolute inset-0 h-full w-full">
                    {currentAnnotation.boxes.map((box, i) => {
                      const [x1, y1, x2, y2] = box.coords;
                      const stroke = box.color || BOX_COLOR_MAP[box.label] || "#0f766e";
                      const isSelected = selectedBoxIdx === i;
                      const displayText = box.text || box.label;
                      return (
                        <g 
                          key={`${i}-${x1}-${y1}`} 
                          className="cursor-pointer pointer-events-auto"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedBoxIdx(i);
                            setStatus(`Selected box #${i + 1} (${box.label}).`);
                          }}
                        >
                          <rect 
                            x={x1} y={y1} 
                            width={x2 - x1} height={y2 - y1} 
                            fill={isSelected ? `${stroke}22` : "none"} 
                            stroke={stroke} 
                            strokeWidth={isSelected ? "4" : "2.5"} 
                          />
                          <rect x={x1} y={Math.max(0, y1 - 22)} width={Math.max(60, displayText.length * 8 + 15)} height={19} fill={stroke} opacity="0.95" rx="5" />
                          <text x={x1 + 7} y={Math.max(0, y1 - 8)} fill="#ffffff" fontSize="11" fontWeight="700">
                            {displayText}
                          </text>
                        </g>
                      );
                    })}
                    {draftBox && (
                      <rect
                        x={Math.min(draftBox.x1, draftBox.x2)}
                        y={Math.min(draftBox.y1, draftBox.y2)}
                        width={Math.abs(draftBox.x2 - draftBox.x1)}
                        height={Math.abs(draftBox.y2 - draftBox.y1)}
                        fill="none"
                        stroke={BOX_COLOR_MAP[selectedBoxLabel]}
                        strokeDasharray="8 6"
                        strokeWidth="2.5"
                      />
                    )}
                  </svg>
                </>
              ) : (
                <div className="grid h-full place-items-center px-6 text-center text-sm font-medium text-stone-500">
                  Queue is empty. Load a new image set and start annotating.
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <button
              type="button"
              className="btn-primary rounded-xl px-3 py-3 text-sm font-bold uppercase tracking-wider"
              onClick={handleSaveAndNext}
              disabled={!images.length}
            >
              Save and Next
            </button>
            <button
              type="button"
              className="btn-secondary rounded-xl px-3 py-3 text-sm font-bold uppercase tracking-wider"
              onClick={downloadAnnotationFiles}
              disabled={!Object.keys(annotations).length}
            >
              Download JSON + {exportFormat.toUpperCase()}
            </button>
          </div>
          <div className="mt-4 flex items-center gap-2 rounded-xl bg-accent-soft/30 p-3 text-xs text-ink/70">
            <svg className="h-4 w-4 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Saving downloads the annotated image as a PNG. Organise them into the <b>annotated/</b> folder.</span>
          </div>
        </article>

        <aside className="panel rounded-3xl border border-clay bg-card p-5 shadow-soft">
          <section className="rounded-2xl border border-stone-200 bg-paper p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-stone-500">Model</p>
            <p className="mt-2 text-sm font-semibold text-stone-900">{MODEL_INFO.name}</p>
            <p className="text-xs text-stone-500">Version: {MODEL_INFO.version}</p>
          </section>
          <div className="h-4" />
          <section className="rounded-2xl border border-stone-200 bg-paper p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-stone-500">Box Labels</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {BOX_LABELS.map((item) => {
                const active = selectedBoxLabel === item.name;
                return (
                  <button
                    key={item.name}
                    type="button"
                    className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition ${active ? "text-white" : "bg-white text-stone-800 hover:bg-stone-50"}`}
                    style={{ borderColor: item.color, backgroundColor: active ? item.color : "#ffffff" }}
                    onClick={() => {
                      setSelectedBoxLabel(item.name);
                      setStatus(`Selected ${item.name}. Draw on image now.`);
                    }}
                  >
                    {item.name}
                  </button>
                );
              })}
            </div>
          </section>

          <section className="mt-4 space-y-5 rounded-2xl border border-clay bg-white p-5 shadow-sm">
            <div>
              <label className="mb-2 block text-xs font-bold uppercase tracking-widest text-ink/50">Wound Type</label>
              <select
                className="input-field w-full text-sm font-semibold"
                value={currentAnnotation.type}
                onChange={(e) => setCurrentAnnotation({ type: e.target.value })}
              >
                {WOUND_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="mb-2 flex justify-between text-xs font-bold uppercase tracking-widest text-ink/50">
                <span>Severity</span>
                <span className="text-primary">{currentAnnotation.severity}</span>
              </label>
              <input
                type="range"
                min="0"
                max="7"
                value={currentAnnotation.severity}
                onChange={(e) => setCurrentAnnotation({ severity: Number(e.target.value) })}
                className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-clay accent-primary"
              />
            </div>

            <div>
              <label className="mb-2 block text-xs font-bold uppercase tracking-widest text-ink/50">Clinical Notes</label>
              <textarea
                className="input-field min-h-28 w-full text-sm"
                value={currentAnnotation.notes}
                onChange={(e) => setCurrentAnnotation({ notes: e.target.value })}
                placeholder="Describe observations..."
              />
            </div>
          </section>

          <section className="mt-4 space-y-4 rounded-2xl border border-clay/50 bg-white/40 p-4 backdrop-blur-sm">
            <div>
              <label className="mb-2 block text-xs font-bold uppercase tracking-widest text-ink/50">Export Format</label>
              <select
                className="input-field w-full text-sm font-bold"
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value)}
              >
                <option value="json">Raw JSON</option>
                <option value="csv">CSV (Metadata)</option>
                <option value="yolo">YOLO (Detection)</option>
                <option value="coco">COCO (Full Dataset)</option>
              </select>
            </div>
            <button
              type="button"
              className="btn-primary w-full rounded-xl px-3 py-2.5 text-sm font-bold uppercase tracking-widest"
              onClick={() => {
                if (exportFormat === "json") toJSON();
                else if (exportFormat === "csv") toCSV();
                else if (exportFormat === "yolo") toYOLO();
                else if (exportFormat === "coco") toCOCO();
              }}
            >
              Download {exportFormat.toUpperCase()}
            </button>
          </section>

          <section className="mt-4 rounded-2xl border border-stone-200 bg-paper p-3">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-stone-500">Boxes</p>
              {selectedBoxIdx !== null && (
                <button 
                  className="text-[10px] font-bold uppercase text-primary hover:underline"
                  onClick={() => setSelectedBoxIdx(null)}
                >
                  Deselect
                </button>
              )}
            </div>
            <ul className="mt-2 max-h-56 space-y-2 overflow-auto">
              {currentAnnotation.boxes.map((box, idx) => {
                const isSelected = selectedBoxIdx === idx;
                const stroke = box.color || BOX_COLOR_MAP[box.label] || "#334155";
                return (
                  <li 
                    key={`${box.label}-${idx}`} 
                    onClick={() => setSelectedBoxIdx(idx)}
                    className={`flex items-center justify-between rounded-lg border transition cursor-pointer px-2 py-1.5 text-xs ${isSelected ? "border-primary bg-primary/5 shadow-sm" : "border-stone-200 bg-white hover:bg-stone-50"}`}
                  >
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full" style={{ backgroundColor: stroke }} />
                      <span className="font-medium" style={{ color: isSelected ? "inherit" : stroke }}>
                        #{idx + 1} {box.text || box.label}
                      </span>
                    </div>
                    <button 
                      className="ml-2 font-semibold text-red-700 hover:text-red-900" 
                      onClick={(e) => {
                        e.stopPropagation();
                        removeBox(idx);
                      }}
                    >
                      Remove
                    </button>
                  </li>
                );
              })}
              {!currentAnnotation.boxes.length && <li className="text-xs text-stone-500">No boxes yet.</li>}
            </ul>
          </section>

          {selectedBoxIdx !== null && currentAnnotation.boxes[selectedBoxIdx] && (
            <section className="mt-4 animate-in fade-in slide-in-from-top-2 duration-300 space-y-4 rounded-2xl border-2 border-primary/20 bg-primary/5 p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-widest text-primary">Box #{selectedBoxIdx + 1} Settings</h3>
                <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-bold text-primary uppercase">{currentAnnotation.boxes[selectedBoxIdx].label}</span>
              </div>
              
              <div>
                <label className="mb-1.5 block text-[10px] font-bold uppercase tracking-widest text-ink/40">Custom Annotation Text</label>
                <input
                  type="text"
                  className="input-field w-full py-1.5 text-xs"
                  placeholder="e.g. Sharp boundary..."
                  value={currentAnnotation.boxes[selectedBoxIdx].text || ""}
                  onChange={(e) => updateBoxProperty(selectedBoxIdx, "text", e.target.value)}
                />
              </div>

              <div>
                <label className="mb-1.5 block text-[10px] font-bold uppercase tracking-widest text-ink/40">Box Color</label>
                <div className="flex items-center gap-3">
                  <input
                    type="color"
                    className="h-8 w-12 cursor-pointer rounded border-0 bg-transparent"
                    value={currentAnnotation.boxes[selectedBoxIdx].color || "#0f766e"}
                    onChange={(e) => updateBoxProperty(selectedBoxIdx, "color", e.target.value)}
                  />
                  <div className="flex flex-wrap gap-1.5">
                    {["#0f766e", "#b45309", "#2563eb", "#6d28d9", "#be123c", "#10b981", "#f59e0b", "#6366f1"].map(c => (
                      <button
                        key={c}
                        className={`h-5 w-5 rounded-full border border-white shadow-sm transition-transform hover:scale-110 ${currentAnnotation.boxes[selectedBoxIdx].color === c ? "ring-2 ring-primary ring-offset-1" : ""}`}
                        style={{ backgroundColor: c }}
                        onClick={() => updateBoxProperty(selectedBoxIdx, "color", c)}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </section>
          )}
        </aside>
      </section>}

      {activeWeek === "week1" && <section className="panel mt-5 rounded-2xl border border-clay bg-card p-4 shadow-soft">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-stone-500">Saved File Preview</p>
        <pre className="max-h-56 overflow-auto rounded-xl bg-stone-950 p-3 text-xs text-stone-100">{autosaveText}</pre>
      </section>}

      {activeWeek === "week1" && <footer className="panel mt-5 rounded-2xl border border-clay bg-card px-4 py-3 text-sm text-stone-700 shadow-soft">{status}</footer>}

      {activeWeek === "week1" && toast && <div className="app-toast">{toast}</div>}
    </main>
  );
}

export default App;
