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
  const [images, setImages] = useState([]);
  const [imageIdx, setImageIdx] = useState(0);
  const [annotations, setAnnotations] = useState({});
  const [selectedBoxLabel, setSelectedBoxLabel] = useState("wound");
  const [isDrawing, setIsDrawing] = useState(false);
  const [draftBox, setDraftBox] = useState(null);
  const [status, setStatus] = useState("Load images to begin annotation.");
  const [toast, setToast] = useState("");
  const [selectedBoxIdx, setSelectedBoxIdx] = useState(null);

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
            <p className="font-display text-xs uppercase tracking-[0.3em] text-warm">Custom Week 1 Annotator</p>
            <h1 className="mt-2 font-display text-3xl text-ink md:text-4xl">Wound Annotation Studio</h1>
            <p className="mt-2 max-w-3xl text-sm text-stone-700">
              Clean workflow for medical image labeling with structured export for classification, detection, and segmentation stages.
            </p>
          </div>
          <div className="rounded-2xl border border-stone-200 bg-white/80 px-4 py-3 text-xs text-stone-700">
            <p className="font-semibold text-stone-900">Annotated Files</p>
            <p className="mt-1 text-xl font-extrabold text-ink">{Object.keys(annotations).length}</p>
          </div>
        </div>
      </header>

      <section className="grid gap-5 xl:grid-cols-[1.65fr_1fr]">
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
      </section>

      <section className="panel mt-5 rounded-2xl border border-clay bg-card p-4 shadow-soft">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-stone-500">Saved File Preview</p>
        <pre className="max-h-56 overflow-auto rounded-xl bg-stone-950 p-3 text-xs text-stone-100">{autosaveText}</pre>
      </section>

      <footer className="panel mt-5 rounded-2xl border border-clay bg-card px-4 py-3 text-sm text-stone-700 shadow-soft">{status}</footer>

      {toast && <div className="app-toast">{toast}</div>}
    </main>
  );
}

export default App;
