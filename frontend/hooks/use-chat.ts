import { useState, useCallback } from "react";
import { useEffect, useRef } from "react";
/* ============================
   TYPES
================================ */

export interface Attachment {
  id: string;
  name: string;
  type: string;
  size: number;
  url: string;
  previewUrl?: string;
  file?: File; // IMPORTANT
}


interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  attachments?: Attachment[];
}

// interface ChatSession {
//   id: string;
//   title: string;
//   createdAt: Date;
//   messages: Message[];
// }
interface ChatSession {
  id: string;
  title: string;
  createdAt: Date;
  messages: Message[];
  pendingGsdRequest?: {
    question: string;
    image: File;
    query_type: string;
  };


  lastImage?: File | null;      // session-specific
  lastPreview?: string | null;  // session-specific
}


/* ============================
   MAIN HOOK
================================ */

export function useChat() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  // const [lastImage, setLastImage] = useState<File | null>(null);
  // const [lastPreview, setLastPreview] = useState<string | null>(null);


    const createdOnce = useRef(false);   // <-- IMPORTANT

  useEffect(() => {
    if (!createdOnce.current && sessions.length === 0) {
      createdOnce.current = true;
      createSession();                 // <-- create only once
    }
  }, [sessions]);




  /* ---------------------------
      BACKEND CALL
  ---------------------------- */
  async function sendVqaRequest(imageFile: File, chat: string) {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("question", chat);
    formData.append("query_type", "auto");

    try {
      const response = await fetch("/api/proxy", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`HTTP ${response.status}:`, errorText);
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error("sendVqaRequest failed:", error);
      throw error;
    }
}


  /* ---------------------------
      SESSION CREATION
  ---------------------------- */
  async function handleNormalBackendResult(result: any, session: ChatSession) {
  let groundingBoxes: any[] = [];

  if (result?.grounding && Array.isArray(result.grounding)) {
    groundingBoxes = result.grounding;
  }

  const imgUrl = session.lastPreview;
  let overlayUrl = null;

  if (imgUrl && groundingBoxes.length > 0) {
    overlayUrl = await drawOrientedBoxesJS(imgUrl, groundingBoxes);
  }

  let assistantText = result?.answer || "";
  if (groundingBoxes.length > 0) assistantText = "Grounded outputs";

  const msg: Message = {
    id: `msg-${Date.now()}`,
    role: "assistant",
    content: assistantText,
    timestamp: new Date(),
    attachments: overlayUrl ? [{
      id: `g-${Date.now()}`,
      name: "grounded.png",
      type: "image/png",
      url: overlayUrl,
      previewUrl: overlayUrl,
      size: 0
    }] : undefined
  };

  setCurrentSession(prev => {
    if (!prev) return prev;
    const updated = { ...prev, messages: [...prev.messages, msg] };
    setSessions(s => s.map(x => x.id === updated.id ? updated : x));
    return updated;
  });
}

async function drawOrientedBoxesJS(imageUrl: string, boxes: any[]) {
  return new Promise<string>((resolve) => {

    if (!boxes || !Array.isArray(boxes) || boxes.length === 0) {
      resolve(imageUrl);
      return;
    }

    const img = new Image();
    img.src = imageUrl;

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;

      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);

      ctx.lineWidth = Math.max(3, Math.floor(Math.min(img.width, img.height) / 200));

      boxes.forEach((b) => {
        if (!b || !b.box || !Array.isArray(b.box) || b.box.length < 4) {
          console.warn("Invalid box:", b);
          return;
        }

        // Model format = [x1, y1, x2, y2, angle] normalized 0–100
        const [x1n, y1n, x2n, y2n, angle = 0] = b.box;

        // Convert normalized → pixel
        const x1 = (x1n / 100) * img.width;
        const y1 = (y1n / 100) * img.height;
        const x2 = (x2n / 100) * img.width;
        const y2 = (y2n / 100) * img.height;

        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        const w = x2 - x1;
        const h = y2 - y1;

        const rad = (angle * Math.PI) / 180;
        const cosA = Math.cos(rad);
        const sinA = Math.sin(rad);

        // Four rectangle corners before rotation
        const corners = [
          [-w/2, -h/2],
          [ w/2, -h/2],
          [ w/2,  h/2],
          [-w/2,  h/2]
        ];

        // Rotate corners
        const rotated = corners.map(([dx, dy]) => ({
          x: cx + dx * cosA - dy * sinA,
          y: cy + dx * sinA + dy * cosA
        }));

        // Draw oriented polygon
        ctx.strokeStyle = "red";
        ctx.beginPath();
        ctx.moveTo(rotated[0].x, rotated[0].y);
        rotated.forEach(pt => ctx.lineTo(pt.x, pt.y));
        ctx.closePath();
        ctx.stroke();
      });

      resolve(canvas.toDataURL("image/png"));
    };
  });
}





  const createSession = useCallback(() => {
    // const newSession: ChatSession = {
    //   id: Date.now().toString(),
    //   title: "New Analysis",
    //   createdAt: new Date(),
    //   messages: [],
    // };
    const newSession: ChatSession = {
     id: Date.now().toString(),
     title: "New Analysis",
     createdAt: new Date(),
     messages: [],
     lastImage: null,
     lastPreview: null,
   };


    setSessions((prev) => [newSession, ...prev]);
    setCurrentSession(newSession);
  }, []);

  /* ---------------------------
      SESSION SELECT / DELETE
  ---------------------------- */

  const selectSession = useCallback(
    (id: string) => {
      const session = sessions.find((s) => s.id === id);
      if (session) setCurrentSession(session);
    },
    [sessions]
  );

  const deleteSession = useCallback(
    (id: string) => {
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (currentSession?.id === id) setCurrentSession(null);
    },
    [currentSession]
  );

  const clearConversations = useCallback(() => {
    setSessions([]);
    setCurrentSession(null);
  }, []);

  /* ---------------------------
      SEND MESSAGE
  ---------------------------- */

  const sendMessage = useCallback(
    async (content: string, attachments?: Attachment[]) => {
      if (!currentSession) {
        createSession();
        return;
      }
      // --------------------------------------------------
// USER IS PROVIDING GSD NUMBER
// --------------------------------------------------
if (currentSession?.pendingGsdRequest &&
    !attachments?.length &&
    !isNaN(Number(content))) {
    
    const gsdValue = Number(content);
    const { question, image, query_type } = currentSession.pendingGsdRequest;

    setCurrentSession(prev => {
        if (!prev) return prev;
        const updated = { ...prev };
        delete updated.pendingGsdRequest;
        setSessions(s => s.map(x => x.id === updated.id ? updated : x));
        return updated;
    });

    const formData = new FormData();
    formData.append("image", image);
    formData.append("question", question);
    formData.append("query_type", query_type);
    formData.append("gsd", String(gsdValue));

    const response = await fetch("/api/proxy", {
        method: "POST",
        body: formData,
    });

    const finalResult = await response.json();

    await handleNormalBackendResult(finalResult, currentSession);

    return;
}



      const userMessage: Message = {
        id: `msg-${Date.now()}`,
        role: "user",
        content,
        timestamp: new Date(),
        attachments: attachments?.length ? attachments : undefined,
      };

      // Add user message
      setCurrentSession((prev) => {
        if (!prev) return prev;
        const updated = {
          ...prev,
          messages: [...prev.messages, userMessage],
        };

        if (updated.messages.length === 1) {
          updated.title =
            content.substring(0, 30) +
            (content.length > 30 ? "..." : "");
        }

        setSessions((sessions) =>
          sessions.map((s) => (s.id === updated.id ? updated : s))
        );

        return updated;
      });

      /* ---------------------------
          SEND TO BACKEND
      ---------------------------- */
      // if (!attachments || !attachments[0] || !attachments[0].file) return;

      setIsLoading(true);
      // Determine which image to send
// // IMAGE SELECTION LOGIC (final clean version)

// If user uploads a NEW image → use it
// -------------------------
// IMAGE SELECTION LOGIC
// -------------------------
// let imageFileToSend: File | null = null;
// let previewToUse: string | null = null;

// // Case 1: user uploaded a new image → use it
// if (attachments?.[0]?.file) {
//     imageFileToSend = attachments[0].file;
//     previewToUse = attachments[0].previewUrl || null;

//     // store for later reuse
//     setLastImage(imageFileToSend);
//     setLastPreview(previewToUse);
// }
// // Case 2: no new image → reuse previous one
// else if (lastImage) {
//     imageFileToSend = lastImage;
//     previewToUse = lastPreview;
// }
// // Case 3: no image at all → block
// else {
//     alert("Please upload an image first!");
//     setIsLoading(false);
//     return;
// }

let imageFileToSend: File | null = null;
let previewToUse: string | null = null;

// Case 1: NEW image uploaded
if (attachments?.[0]?.file) {
    imageFileToSend = attachments[0].file;
    previewToUse = attachments[0].previewUrl || null;

    // Save inside session
    setCurrentSession(prev => {
        if (!prev) return prev;
        const updated = {
            ...prev,
            lastImage: imageFileToSend,
            lastPreview: previewToUse,
        };
        setSessions(s => s.map(x => x.id === updated.id ? updated : x));
        return updated;
    });
}
// Case 2: REUSE session image
else if (currentSession?.lastImage) {
    imageFileToSend = currentSession.lastImage;
    previewToUse = currentSession.lastPreview || null;
}
// Case 3: No image at all
else {
    alert("Please upload an image first!");
    setIsLoading(false);
    return;
}



      try {
        // const result = await sendVqaRequest(attachments[0].file, content);
        const result = await sendVqaRequest(imageFileToSend, content);
        if (result?._type === "missing_gsd") {

  // Store original question + image
  setCurrentSession(prev => {
    if (!prev) return prev;
    const updated = {
      ...prev,
     pendingGsdRequest: {
    question: content,
    image: imageFileToSend!,
    query_type: "auto",
    }

    };
    setSessions(s => s.map(x => x.id === updated.id ? updated : x));
    return updated;
  });

  // Ask user to provide GSD
  const assistantMsg: Message = {
    id: `msg-${Date.now()}`,
    role: "assistant",
    content: result.answer || "Please enter GSD value.",
    timestamp: new Date(),
  };

  setCurrentSession(prev => {
    if (!prev) return prev;
    const updated = {
      ...prev,
      messages: [...prev.messages, assistantMsg],
    };
    setSessions(s => s.map(x => x.id === updated.id ? updated : x));
    return updated;
  });

  setIsLoading(false);
  return;
}


        console.log("Backend response:", result);
        // alert(JSON.stringify(result, null, 2));

      let groundingBoxes: any[] = [];

function pythonListToJson(str: string) {
  return str
    .replace(/'/g, '"')                         // Convert single → double quotes
    .replace(/(\w+):/g, '"$1":');               // Convert label: → "label":
}

if (result?.grounding && Array.isArray(result.grounding)) {
  groundingBoxes = result.grounding;
} 
else if (result?.raw_output) {
  try {
    const raw = result.raw_output;

    // Extract python list
    const match = raw.match(/\[[\s\S]*\]/);
    if (match) {
      const pythonList = match[0];
      const jsonReady = pythonListToJson(pythonList);

      const parsed = JSON.parse(jsonReady);

      groundingBoxes = parsed.map((b: any) => ({
        label: b.label || "detected_object",
        box: b.box
      }));
    }
  } catch (err) {
    console.error("Grounding parsing failed:", err);
  }
}

        // const originalImageUrl = attachments?.[0]?.previewUrl;
        // const originalImageUrl = (attachments && attachments[0]?.previewUrl) || lastPreview;
        const originalImageUrl =
  (attachments && attachments[0]?.previewUrl) || currentSession?.lastPreview || null;

        
  

        let groundedImageUrl: string | null = null;
        
        if (groundingBoxes && groundingBoxes.length > 0 && originalImageUrl) {
            groundedImageUrl = await drawOrientedBoxesJS(originalImageUrl, groundingBoxes);
        }


        let assistantText = result?.answer || "";

        // If this query returned grounding boxes → hide ugly JSON output
         if (groundingBoxes && groundingBoxes.length > 0) {
           assistantText = "Grounded outputs";
         }


const assistantMessage: Message = {
  id: `msg-${Date.now()}`,
  role: "assistant",
  content: assistantText.trim(),
  timestamp: new Date(),
  attachments: groundedImageUrl
    ? [
        {
          id: `grounded-${Date.now()}`,
          name: "Grounded",
          type: "image/png",
          size: 0,
          url: groundedImageUrl,
          previewUrl: groundedImageUrl,
        },
      ]
    : undefined,
};

        setCurrentSession((prev) => {
          if (!prev) return prev;
          const updated = {
            ...prev,
            messages: [...prev.messages, assistantMessage],
          };
          setSessions((sessions) =>
            sessions.map((s) => (s.id === updated.id ? updated : s))
          );
          return updated;
        });
      } catch (e) {
        console.error("Backend error:", e);
      }

      setIsLoading(false);
    },
    [currentSession, createSession]
  );

  return {
    sessions,
    currentSession,
    messages: currentSession?.messages || [],
    isLoading,
    createSession,
    selectSession,
    deleteSession,
    clearConversations,
    sendMessage,
  };
}
