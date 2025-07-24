package com.google.ar.sceneform;

import android.view.MotionEvent;
import com.google.ar.sceneform.Scene;
import com.google.ar.sceneform.utilities.Preconditions;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Iterator;

/* loaded from: classes.dex */
public class TouchEventSystem {
    private Method motionEventSplitMethod;
    private Scene.OnTouchListener onTouchListener;
    private final Object[] motionEventSplitParams = new Object[1];
    private final ArrayList<Scene.OnPeekTouchListener> onPeekTouchListeners = new ArrayList<>();
    private Scene.OnTouchListener handlingTouchListener = null;
    private TouchTarget firstHandlingTouchTarget = null;

    /* loaded from: classes.dex */
    public static class TouchTarget {
        public static final int ALL_POINTER_IDS = -1;
        public TouchTarget next;
        public Node node;
        public int pointerIdBits;

        private TouchTarget() {
        }
    }

    private TouchTarget addTouchTarget(Node node, int i) {
        TouchTarget touchTarget = new TouchTarget();
        touchTarget.node = node;
        touchTarget.pointerIdBits = i;
        touchTarget.next = this.firstHandlingTouchTarget;
        this.firstHandlingTouchTarget = touchTarget;
        return touchTarget;
    }

    private void clearTouchTargets() {
        this.handlingTouchListener = null;
        this.firstHandlingTouchTarget = null;
    }

    private Node dispatchTouchEvent(MotionEvent motionEvent, HitTestResult hitTestResult, Node node, int i, boolean z) {
        int pointerIdBits = getPointerIdBits(motionEvent);
        int i2 = i & pointerIdBits;
        if (i2 == 0) {
            return null;
        }
        boolean z2 = false;
        if (i2 != pointerIdBits) {
            motionEvent = splitMotionEvent(motionEvent, i2);
            z2 = true;
        }
        while (node != null && !node.dispatchTouchEvent(hitTestResult, motionEvent)) {
            node = z ? node.getParent() : null;
        }
        if (node == null) {
            tryDispatchToSceneTouchListener(hitTestResult, motionEvent);
        }
        if (z2) {
            motionEvent.recycle();
        }
        return node;
    }

    private int getPointerIdBits(MotionEvent motionEvent) {
        int pointerCount = motionEvent.getPointerCount();
        int i = 0;
        for (int i2 = 0; i2 < pointerCount; i2++) {
            i |= 1 << motionEvent.getPointerId(i2);
        }
        return i;
    }

    private TouchTarget getTouchTargetForNode(Node node) {
        for (TouchTarget touchTarget = this.firstHandlingTouchTarget; touchTarget != null; touchTarget = touchTarget.next) {
            if (touchTarget.node == node) {
                return touchTarget;
            }
        }
        return null;
    }

    private void removePointersFromTouchTargets(int i) {
        TouchTarget touchTarget = this.firstHandlingTouchTarget;
        TouchTarget touchTarget2 = null;
        while (touchTarget != null) {
            TouchTarget touchTarget3 = touchTarget.next;
            int i2 = touchTarget.pointerIdBits;
            if ((i2 & i) != 0) {
                int i3 = i2 & (~i);
                touchTarget.pointerIdBits = i3;
                if (i3 == 0) {
                    if (touchTarget2 == null) {
                        this.firstHandlingTouchTarget = touchTarget3;
                    } else {
                        touchTarget2.next = touchTarget3;
                    }
                    touchTarget = touchTarget3;
                }
            }
            touchTarget2 = touchTarget;
            touchTarget = touchTarget3;
        }
    }

    private MotionEvent splitMotionEvent(MotionEvent motionEvent, int i) {
        if (this.motionEventSplitMethod == null) {
            try {
                this.motionEventSplitMethod = MotionEvent.class.getMethod("split", Integer.TYPE);
            } catch (ReflectiveOperationException e2) {
                throw new RuntimeException("Splitting MotionEvent not supported.", e2);
            }
        }
        try {
            this.motionEventSplitParams[0] = Integer.valueOf(i);
            Object invoke = this.motionEventSplitMethod.invoke(motionEvent, this.motionEventSplitParams);
            return invoke != null ? (MotionEvent) invoke : motionEvent;
        } catch (IllegalAccessException | InvocationTargetException e3) {
            throw new RuntimeException("Unable to split MotionEvent.", e3);
        }
    }

    private boolean tryDispatchToSceneTouchListener(HitTestResult hitTestResult, MotionEvent motionEvent) {
        if (motionEvent.getActionMasked() == 0) {
            Scene.OnTouchListener onTouchListener = this.onTouchListener;
            if (onTouchListener == null || !onTouchListener.onSceneTouch(hitTestResult, motionEvent)) {
                return false;
            }
            this.handlingTouchListener = this.onTouchListener;
            return true;
        }
        Scene.OnTouchListener onTouchListener2 = this.handlingTouchListener;
        if (onTouchListener2 != null) {
            onTouchListener2.onSceneTouch(hitTestResult, motionEvent);
            return true;
        }
        return false;
    }

    public void addOnPeekTouchListener(Scene.OnPeekTouchListener onPeekTouchListener) {
        if (this.onPeekTouchListeners.contains(onPeekTouchListener)) {
            return;
        }
        this.onPeekTouchListeners.add(onPeekTouchListener);
    }

    public Scene.OnTouchListener getOnTouchListener() {
        return this.onTouchListener;
    }

    public void onTouchEvent(HitTestResult hitTestResult, MotionEvent motionEvent) {
        boolean z;
        TouchTarget touchTarget;
        Preconditions.checkNotNull(hitTestResult, "Parameter \"hitTestResult\" was null.");
        Preconditions.checkNotNull(motionEvent, "Parameter \"motionEvent\" was null.");
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 0) {
            clearTouchTargets();
        }
        Iterator<Scene.OnPeekTouchListener> it = this.onPeekTouchListeners.iterator();
        while (it.hasNext()) {
            it.next().onPeekTouch(hitTestResult, motionEvent);
        }
        if (this.handlingTouchListener != null) {
            tryDispatchToSceneTouchListener(hitTestResult, motionEvent);
        } else {
            TouchTarget touchTarget2 = null;
            Node node = hitTestResult.getNode();
            boolean z2 = false;
            if (actionMasked == 0 || actionMasked == 5) {
                int pointerId = 1 << motionEvent.getPointerId(motionEvent.getActionIndex());
                removePointersFromTouchTargets(pointerId);
                if (node != null) {
                    touchTarget2 = getTouchTargetForNode(node);
                    if (touchTarget2 != null) {
                        touchTarget2.pointerIdBits |= pointerId;
                    } else {
                        Node dispatchTouchEvent = dispatchTouchEvent(motionEvent, hitTestResult, node, pointerId, true);
                        if (dispatchTouchEvent != null) {
                            touchTarget2 = addTouchTarget(dispatchTouchEvent, pointerId);
                            z2 = true;
                        }
                        z = z2;
                        z2 = true;
                        if (touchTarget2 == null && (touchTarget = this.firstHandlingTouchTarget) != null) {
                            do {
                                touchTarget2 = touchTarget;
                                touchTarget = touchTarget2.next;
                            } while (touchTarget != null);
                            touchTarget2.pointerIdBits |= pointerId;
                        }
                    }
                }
                z = false;
                if (touchTarget2 == null) {
                    do {
                        touchTarget2 = touchTarget;
                        touchTarget = touchTarget2.next;
                    } while (touchTarget != null);
                    touchTarget2.pointerIdBits |= pointerId;
                }
            } else {
                z = false;
            }
            TouchTarget touchTarget3 = this.firstHandlingTouchTarget;
            if (touchTarget3 != null) {
                while (touchTarget3 != null) {
                    TouchTarget touchTarget4 = touchTarget3.next;
                    if (!z || touchTarget3 != touchTarget2) {
                        dispatchTouchEvent(motionEvent, hitTestResult, touchTarget3.node, touchTarget3.pointerIdBits, false);
                    }
                    touchTarget3 = touchTarget4;
                }
            } else if (!z2) {
                tryDispatchToSceneTouchListener(hitTestResult, motionEvent);
            }
        }
        if (actionMasked == 3 || actionMasked == 1) {
            clearTouchTargets();
        } else if (actionMasked == 6) {
            removePointersFromTouchTargets(1 << motionEvent.getPointerId(motionEvent.getActionIndex()));
        }
    }

    public void removeOnPeekTouchListener(Scene.OnPeekTouchListener onPeekTouchListener) {
        this.onPeekTouchListeners.remove(onPeekTouchListener);
    }

    public void setOnTouchListener(Scene.OnTouchListener onTouchListener) {
        this.onTouchListener = onTouchListener;
    }
}