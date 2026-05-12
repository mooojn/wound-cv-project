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

    const nextBox = { coords: [x1, y1, x2, y2], label: selectedBoxLabel };
    setCurrentAnnotation({ boxes: [...currentAnnotation.boxes, nextBox] });
    setDraftBox(null);
    setStatus(`Added ${selectedBoxLabel} box.`);
  };

  const removeBox = (idx) => {
    const nextBoxes = currentAnnotation.boxes.filter((_, i) => i !== idx);
    setCurrentAnnotation({ boxes: nextBoxes });
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

  const handleSaveAndNext = () => {
    const saved = persistCurrentAnnotation();
    if (!saved) {
      setStatus("Load images first.");
      return;
    }

    const nextSnapshot = { ...annotations, [saved.imageName]: saved.annotation };
    saveAutosaveFile(nextSnapshot);

    setImages((prev) => {
      const next = prev.filter((_, idx) => idx !== imageIdx);
      URL.revokeObjectURL(prev[imageIdx].url);
      return next;
    });
    setImageIdx(0);

    setToast(`Saved ${saved.imageName} successfully`);
    setStatus("Saved and removed current image from queue. Load new images or continue.");
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
    const categories = WOUND_TYPES.map((name, id) => ({ id, name }));
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
          category_id: WOUND_TYPES.indexOf(data.type),
          bbox: [x1, y1, x2 - x1, y2 - y1],
          area: (x2 - x1) * (y2 - y1),
          iscrowd: 0,
          metadata: { box_label: box.label, severity: data.severity, notes: data.notes ?? "" },
        });
        annId += 1;
      });
    });

    const payload = { images: imagesOut, annotations: annotationsOut, categories, box_labels: BOX_LABELS };
    downloadFile("annotations.json", "application/json;charset=utf-8", JSON.stringify(payload, null, 2));
  };

  const downloadAnnotationFiles = () => {
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

    const rawJson = localStorage.getItem("week1_annotations_file") || autosaveText;
    downloadFile("annotations_autosave.json", "application/json;charset=utf-8", rawJson, false);
    downloadFile("annotations_autosave.csv", "text/csv;charset=utf-8", csv, false);
    setStatus("Downloaded annotation JSON and CSV files.");
    setToast("Annotation JSON and CSV downloaded");
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
            <label className="btn-dark inline-flex cursor-pointer items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold text-paper">
              Load Images
              <input type="file" accept="image/*" multiple className="hidden" onChange={loadImages} />
            </label>
            <label className="btn-light inline-flex cursor-pointer items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold text-stone-800">
              Load Folder
              <input type="file" accept="image/*" multiple className="hidden" webkitdirectory="" directory="" onChange={loadImages} />
            </label>
            <div className="ml-auto rounded-xl border border-stone-200 bg-white px-3 py-2 text-xs font-semibold text-stone-700">
              {currentImage ? `${imageIdx + 1}/${images.length}  ${currentImage.name}` : "No image selected"}
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
                      const stroke = BOX_COLOR_MAP[box.label] ?? "#0f766e";
                      return (
                        <g key={`${i}-${x1}-${y1}`}>
                          <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={stroke} strokeWidth="2.5" />
                          <rect x={x1} y={Math.max(0, y1 - 22)} width={95} height={19} fill={stroke} opacity="0.95" rx="5" />
                          <text x={x1 + 7} y={Math.max(0, y1 - 8)} fill="#ffffff" fontSize="11" fontWeight="700">
                            {box.label}
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

          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            <button
              type="button"
              className="btn-accent rounded-xl px-3 py-2.5 text-sm font-semibold text-white"
              onClick={handleSaveAndNext}
              disabled={!images.length}
            >
              Save and Next
            </button>
            <button type="button" className="btn-light rounded-xl px-3 py-2.5 text-sm font-semibold" onClick={downloadAnnotationFiles}>
              Download JSON + CSV
            </button>
          </div>
        </article>

        <aside className="panel rounded-3xl border border-clay bg-card p-5 shadow-soft">
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

          <section className="mt-4 space-y-4 rounded-2xl border border-stone-200 bg-white p-4">
            <div>
              <label className="mb-1 block text-sm font-semibold text-stone-700">Wound Type</label>
              <select
                className="w-full rounded-xl border border-stone-300 bg-white px-3 py-2 text-sm"
                value={currentAnnotation.type}
                onChange={(e) => setCurrentAnnotation({ type: e.target.value })}
              >
                {WOUND_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="mb-1 block text-sm font-semibold text-stone-700">Severity: {currentAnnotation.severity}</label>
              <input
                type="range"
                min="0"
                max="7"
                value={currentAnnotation.severity}
                onChange={(e) => setCurrentAnnotation({ severity: Number(e.target.value) })}
                className="w-full accent-accent"
              />
            </div>

            <div>
              <label className="mb-1 block text-sm font-semibold text-stone-700">Clinical Notes</label>
              <textarea
                className="min-h-28 w-full rounded-xl border border-stone-300 bg-white px-3 py-2 text-sm"
                value={currentAnnotation.notes}
                onChange={(e) => setCurrentAnnotation({ notes: e.target.value })}
                placeholder="Add observations for this image"
              />
            </div>
          </section>

          <section className="mt-4 grid gap-2 sm:grid-cols-2">
            <button type="button" className="btn-dark rounded-xl px-3 py-2.5 text-sm font-semibold text-paper" onClick={toCSV}>
              Export CSV
            </button>
            <button type="button" className="btn-warm rounded-xl px-3 py-2.5 text-sm font-semibold text-white" onClick={toJSON}>
              Export JSON
            </button>
          </section>

          <section className="mt-4 rounded-2xl border border-stone-200 bg-paper p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-stone-500">Boxes</p>
            <ul className="mt-2 max-h-56 space-y-2 overflow-auto">
              {currentAnnotation.boxes.map((box, idx) => (
                <li key={`${box.label}-${idx}`} className="flex items-center justify-between rounded-lg border border-stone-200 bg-white px-2 py-1.5 text-xs">
                  <span className="font-medium" style={{ color: BOX_COLOR_MAP[box.label] ?? "#334155" }}>
                    #{idx + 1} {box.label}
                  </span>
                  <button className="font-semibold text-red-700" onClick={() => removeBox(idx)}>
                    Remove
                  </button>
                </li>
              ))}
              {!currentAnnotation.boxes.length && <li className="text-xs text-stone-500">No boxes yet.</li>}
            </ul>
          </section>
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
