package b.l.b;

import android.graphics.Rect;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewParent;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityManager;
import android.view.accessibility.AccessibilityNodeInfo;
import b.f.i;
import b.j.j.q;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.calib3d.Calib3d;

/* compiled from: ExploreByTouchHelper.java */
/* loaded from: classes.dex */
public abstract class a extends b.j.j.a {
    private static final String DEFAULT_CLASS_NAME = "android.view.View";
    public static final int HOST_ID = -1;
    public static final int INVALID_ID = Integer.MIN_VALUE;
    private static final Rect INVALID_PARENT_BOUNDS = new Rect(Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE);
    private static final b.l.b.b<b.j.j.x.b> NODE_ADAPTER = new C0042a();
    private static final b.l.b.c<i<b.j.j.x.b>, b.j.j.x.b> SPARSE_VALUES_ADAPTER = new b();
    private final View mHost;
    private final AccessibilityManager mManager;
    private c mNodeProvider;
    private final Rect mTempScreenRect = new Rect();
    private final Rect mTempParentRect = new Rect();
    private final Rect mTempVisibleRect = new Rect();
    private final int[] mTempGlobalRect = new int[2];
    public int mAccessibilityFocusedVirtualViewId = Integer.MIN_VALUE;
    public int mKeyboardFocusedVirtualViewId = Integer.MIN_VALUE;
    private int mHoveredVirtualViewId = Integer.MIN_VALUE;

    /* compiled from: ExploreByTouchHelper.java */
    /* renamed from: b.l.b.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0042a implements b.l.b.b<b.j.j.x.b> {
        public void a(Object obj, Rect rect) {
            ((b.j.j.x.b) obj).f2259b.getBoundsInParent(rect);
        }
    }

    /* compiled from: ExploreByTouchHelper.java */
    /* loaded from: classes.dex */
    public class b implements b.l.b.c<i<b.j.j.x.b>, b.j.j.x.b> {
    }

    /* compiled from: ExploreByTouchHelper.java */
    /* loaded from: classes.dex */
    public class c extends b.j.j.x.c {
        public c() {
        }

        @Override // b.j.j.x.c
        public b.j.j.x.b a(int i) {
            return new b.j.j.x.b(AccessibilityNodeInfo.obtain(a.this.obtainAccessibilityNodeInfo(i).f2259b));
        }

        @Override // b.j.j.x.c
        public b.j.j.x.b b(int i) {
            int i2 = i == 2 ? a.this.mAccessibilityFocusedVirtualViewId : a.this.mKeyboardFocusedVirtualViewId;
            if (i2 == Integer.MIN_VALUE) {
                return null;
            }
            return new b.j.j.x.b(AccessibilityNodeInfo.obtain(a.this.obtainAccessibilityNodeInfo(i2).f2259b));
        }

        @Override // b.j.j.x.c
        public boolean c(int i, int i2, Bundle bundle) {
            return a.this.performAction(i, i2, bundle);
        }
    }

    public a(View view) {
        if (view != null) {
            this.mHost = view;
            this.mManager = (AccessibilityManager) view.getContext().getSystemService("accessibility");
            view.setFocusable(true);
            AtomicInteger atomicInteger = q.f2214a;
            if (view.getImportantForAccessibility() == 0) {
                view.setImportantForAccessibility(1);
                return;
            }
            return;
        }
        throw new IllegalArgumentException("View may not be null");
    }

    private boolean clearAccessibilityFocus(int i) {
        if (this.mAccessibilityFocusedVirtualViewId == i) {
            this.mAccessibilityFocusedVirtualViewId = Integer.MIN_VALUE;
            this.mHost.invalidate();
            sendEventForVirtualView(i, 65536);
            return true;
        }
        return false;
    }

    private boolean clickKeyboardFocusedVirtualView() {
        int i = this.mKeyboardFocusedVirtualViewId;
        return i != Integer.MIN_VALUE && onPerformActionForVirtualView(i, 16, null);
    }

    private AccessibilityEvent createEvent(int i, int i2) {
        if (i != -1) {
            return createEventForChild(i, i2);
        }
        return createEventForHost(i2);
    }

    private AccessibilityEvent createEventForChild(int i, int i2) {
        AccessibilityEvent obtain = AccessibilityEvent.obtain(i2);
        b.j.j.x.b obtainAccessibilityNodeInfo = obtainAccessibilityNodeInfo(i);
        obtain.getText().add(obtainAccessibilityNodeInfo.i());
        obtain.setContentDescription(obtainAccessibilityNodeInfo.g());
        obtain.setScrollable(obtainAccessibilityNodeInfo.f2259b.isScrollable());
        obtain.setPassword(obtainAccessibilityNodeInfo.f2259b.isPassword());
        obtain.setEnabled(obtainAccessibilityNodeInfo.j());
        obtain.setChecked(obtainAccessibilityNodeInfo.f2259b.isChecked());
        onPopulateEventForVirtualView(i, obtain);
        if (obtain.getText().isEmpty() && obtain.getContentDescription() == null) {
            throw new RuntimeException("Callbacks must add text or a content description in populateEventForVirtualViewId()");
        }
        obtain.setClassName(obtainAccessibilityNodeInfo.e());
        obtain.setSource(this.mHost, i);
        obtain.setPackageName(this.mHost.getContext().getPackageName());
        return obtain;
    }

    private AccessibilityEvent createEventForHost(int i) {
        AccessibilityEvent obtain = AccessibilityEvent.obtain(i);
        this.mHost.onInitializeAccessibilityEvent(obtain);
        return obtain;
    }

    private b.j.j.x.b createNodeForChild(int i) {
        AccessibilityNodeInfo obtain = AccessibilityNodeInfo.obtain();
        b.j.j.x.b bVar = new b.j.j.x.b(obtain);
        obtain.setEnabled(true);
        obtain.setFocusable(true);
        obtain.setClassName(DEFAULT_CLASS_NAME);
        Rect rect = INVALID_PARENT_BOUNDS;
        obtain.setBoundsInParent(rect);
        obtain.setBoundsInScreen(rect);
        bVar.p(this.mHost);
        onPopulateNodeForVirtualView(i, bVar);
        if (bVar.i() == null && bVar.g() == null) {
            throw new RuntimeException("Callbacks must add text or a content description in populateNodeForVirtualViewId()");
        }
        obtain.getBoundsInParent(this.mTempParentRect);
        if (!this.mTempParentRect.equals(rect)) {
            int d2 = bVar.d();
            if ((d2 & 64) == 0) {
                if ((d2 & 128) == 0) {
                    obtain.setPackageName(this.mHost.getContext().getPackageName());
                    View view = this.mHost;
                    bVar.f2261d = i;
                    obtain.setSource(view, i);
                    if (this.mAccessibilityFocusedVirtualViewId == i) {
                        obtain.setAccessibilityFocused(true);
                        obtain.addAction(128);
                    } else {
                        obtain.setAccessibilityFocused(false);
                        obtain.addAction(64);
                    }
                    boolean z = this.mKeyboardFocusedVirtualViewId == i;
                    if (z) {
                        obtain.addAction(2);
                    } else if (obtain.isFocusable()) {
                        obtain.addAction(1);
                    }
                    obtain.setFocused(z);
                    this.mHost.getLocationOnScreen(this.mTempGlobalRect);
                    obtain.getBoundsInScreen(this.mTempScreenRect);
                    if (this.mTempScreenRect.equals(rect)) {
                        obtain.getBoundsInParent(this.mTempScreenRect);
                        if (bVar.f2260c != -1) {
                            b.j.j.x.b bVar2 = new b.j.j.x.b(AccessibilityNodeInfo.obtain());
                            for (int i2 = bVar.f2260c; i2 != -1; i2 = bVar2.f2260c) {
                                View view2 = this.mHost;
                                bVar2.f2260c = -1;
                                bVar2.f2259b.setParent(view2, -1);
                                bVar2.f2259b.setBoundsInParent(INVALID_PARENT_BOUNDS);
                                onPopulateNodeForVirtualView(i2, bVar2);
                                bVar2.f2259b.getBoundsInParent(this.mTempParentRect);
                                Rect rect2 = this.mTempScreenRect;
                                Rect rect3 = this.mTempParentRect;
                                rect2.offset(rect3.left, rect3.top);
                            }
                            bVar2.f2259b.recycle();
                        }
                        this.mTempScreenRect.offset(this.mTempGlobalRect[0] - this.mHost.getScrollX(), this.mTempGlobalRect[1] - this.mHost.getScrollY());
                    }
                    if (this.mHost.getLocalVisibleRect(this.mTempVisibleRect)) {
                        this.mTempVisibleRect.offset(this.mTempGlobalRect[0] - this.mHost.getScrollX(), this.mTempGlobalRect[1] - this.mHost.getScrollY());
                        if (this.mTempScreenRect.intersect(this.mTempVisibleRect)) {
                            bVar.f2259b.setBoundsInScreen(this.mTempScreenRect);
                            if (isVisibleToUser(this.mTempScreenRect)) {
                                bVar.f2259b.setVisibleToUser(true);
                            }
                        }
                    }
                    return bVar;
                }
                throw new RuntimeException("Callbacks must not add ACTION_CLEAR_ACCESSIBILITY_FOCUS in populateNodeForVirtualViewId()");
            }
            throw new RuntimeException("Callbacks must not add ACTION_ACCESSIBILITY_FOCUS in populateNodeForVirtualViewId()");
        }
        throw new RuntimeException("Callbacks must set parent bounds in populateNodeForVirtualViewId()");
    }

    private b.j.j.x.b createNodeForHost() {
        AccessibilityNodeInfo obtain = AccessibilityNodeInfo.obtain(this.mHost);
        b.j.j.x.b bVar = new b.j.j.x.b(obtain);
        View view = this.mHost;
        AtomicInteger atomicInteger = q.f2214a;
        view.onInitializeAccessibilityNodeInfo(obtain);
        ArrayList arrayList = new ArrayList();
        getVisibleVirtualViews(arrayList);
        if (obtain.getChildCount() > 0 && arrayList.size() > 0) {
            throw new RuntimeException("Views cannot have both real and virtual children");
        }
        int size = arrayList.size();
        for (int i = 0; i < size; i++) {
            bVar.f2259b.addChild(this.mHost, ((Integer) arrayList.get(i)).intValue());
        }
        return bVar;
    }

    private i<b.j.j.x.b> getAllNodes() {
        ArrayList arrayList = new ArrayList();
        getVisibleVirtualViews(arrayList);
        i<b.j.j.x.b> iVar = new i<>(10);
        for (int i = 0; i < arrayList.size(); i++) {
            iVar.g(((Integer) arrayList.get(i)).intValue(), createNodeForChild(((Integer) arrayList.get(i)).intValue()));
        }
        return iVar;
    }

    private void getBoundsInParent(int i, Rect rect) {
        obtainAccessibilityNodeInfo(i).f2259b.getBoundsInParent(rect);
    }

    private static Rect guessPreviouslyFocusedRect(View view, int i, Rect rect) {
        int width = view.getWidth();
        int height = view.getHeight();
        if (i == 17) {
            rect.set(width, 0, width, height);
        } else if (i == 33) {
            rect.set(0, height, width, height);
        } else if (i == 66) {
            rect.set(-1, 0, -1, height);
        } else if (i == 130) {
            rect.set(0, -1, width, -1);
        } else {
            throw new IllegalArgumentException("direction must be one of {FOCUS_UP, FOCUS_DOWN, FOCUS_LEFT, FOCUS_RIGHT}.");
        }
        return rect;
    }

    private boolean isVisibleToUser(Rect rect) {
        if (rect == null || rect.isEmpty() || this.mHost.getWindowVisibility() != 0) {
            return false;
        }
        ViewParent parent = this.mHost.getParent();
        while (parent instanceof View) {
            View view = (View) parent;
            if (view.getAlpha() <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || view.getVisibility() != 0) {
                return false;
            }
            parent = view.getParent();
        }
        return parent != null;
    }

    private static int keyToDirection(int i) {
        if (i != 19) {
            if (i != 21) {
                return i != 22 ? 130 : 66;
            }
            return 17;
        }
        return 33;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r5v5, resolved type: java.lang.Object */
    /* JADX WARN: Code restructure failed: missing block: B:55:0x00f4, code lost:
        if (r13 < ((r18 * r18) + ((r17 * 13) * r17))) goto L41;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:109:0x0100 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:59:0x00fb  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private boolean moveFocus(int i, Rect rect) {
        Object obj;
        b.j.j.x.b bVar;
        int f2;
        boolean z;
        i<b.j.j.x.b> allNodes = getAllNodes();
        int i2 = this.mKeyboardFocusedVirtualViewId;
        b.j.j.x.b d2 = i2 == Integer.MIN_VALUE ? null : allNodes.d(i2);
        int i3 = -1;
        int i4 = 0;
        if (i == 1 || i == 2) {
            View view = this.mHost;
            AtomicInteger atomicInteger = q.f2214a;
            boolean z2 = view.getLayoutDirection() == 1;
            b.l.b.c<i<b.j.j.x.b>, b.j.j.x.b> cVar = SPARSE_VALUES_ADAPTER;
            b.l.b.b<b.j.j.x.b> bVar2 = NODE_ADAPTER;
            Objects.requireNonNull((b) cVar);
            int i5 = allNodes.i();
            ArrayList arrayList = new ArrayList(i5);
            for (int i6 = 0; i6 < i5; i6++) {
                if (allNodes.f1777c) {
                    allNodes.c();
                }
                arrayList.add((b.j.j.x.b) allNodes.f1779e[i6]);
            }
            Collections.sort(arrayList, new d(z2, bVar2));
            if (i == 1) {
                int size = arrayList.size();
                if (d2 != null) {
                    size = arrayList.indexOf(d2);
                }
                int i7 = size - 1;
                if (i7 >= 0) {
                    obj = arrayList.get(i7);
                    bVar = (b.j.j.x.b) obj;
                }
                obj = null;
                bVar = (b.j.j.x.b) obj;
            } else if (i == 2) {
                int size2 = arrayList.size();
                int lastIndexOf = (d2 == null ? -1 : arrayList.lastIndexOf(d2)) + 1;
                if (lastIndexOf < size2) {
                    obj = arrayList.get(lastIndexOf);
                    bVar = (b.j.j.x.b) obj;
                }
                obj = null;
                bVar = (b.j.j.x.b) obj;
            } else {
                throw new IllegalArgumentException("direction must be one of {FOCUS_FORWARD, FOCUS_BACKWARD}.");
            }
        } else if (i != 17 && i != 33 && i != 66 && i != 130) {
            throw new IllegalArgumentException("direction must be one of {FOCUS_FORWARD, FOCUS_BACKWARD, FOCUS_UP, FOCUS_DOWN, FOCUS_LEFT, FOCUS_RIGHT}.");
        } else {
            Rect rect2 = new Rect();
            int i8 = this.mKeyboardFocusedVirtualViewId;
            if (i8 != Integer.MIN_VALUE) {
                getBoundsInParent(i8, rect2);
            } else if (rect != null) {
                rect2.set(rect);
            } else {
                guessPreviouslyFocusedRect(this.mHost, i, rect2);
            }
            b.l.b.c<i<b.j.j.x.b>, b.j.j.x.b> cVar2 = SPARSE_VALUES_ADAPTER;
            b.l.b.b<b.j.j.x.b> bVar3 = NODE_ADAPTER;
            Rect rect3 = new Rect(rect2);
            if (i == 17) {
                rect3.offset(rect2.width() + 1, 0);
            } else if (i == 33) {
                rect3.offset(0, rect2.height() + 1);
            } else if (i == 66) {
                rect3.offset(-(rect2.width() + 1), 0);
            } else if (i == 130) {
                rect3.offset(0, -(rect2.height() + 1));
            } else {
                throw new IllegalArgumentException("direction must be one of {FOCUS_UP, FOCUS_DOWN, FOCUS_LEFT, FOCUS_RIGHT}.");
            }
            Objects.requireNonNull((b) cVar2);
            int i9 = allNodes.i();
            Rect rect4 = new Rect();
            bVar = null;
            for (int i10 = 0; i10 < i9; i10++) {
                if (allNodes.f1777c) {
                    allNodes.c();
                }
                b.j.j.x.b bVar4 = (b.j.j.x.b) allNodes.f1779e[i10];
                if (bVar4 != d2) {
                    ((C0042a) bVar3).a(bVar4, rect4);
                    if (b.j.b.d.z(rect2, rect4, i)) {
                        if (b.j.b.d.z(rect2, rect3, i) && !b.j.b.d.a(i, rect2, rect4, rect3)) {
                            if (!b.j.b.d.a(i, rect2, rect3, rect4)) {
                                int C = b.j.b.d.C(i, rect2, rect4);
                                int E = b.j.b.d.E(i, rect2, rect4);
                                int i11 = (E * E) + (C * 13 * C);
                                int C2 = b.j.b.d.C(i, rect2, rect3);
                                int E2 = b.j.b.d.E(i, rect2, rect3);
                            }
                        }
                        z = true;
                        if (!z) {
                            rect3.set(rect4);
                            bVar = bVar4;
                        }
                    }
                    z = false;
                    if (!z) {
                    }
                }
            }
        }
        b.j.j.x.b bVar5 = bVar;
        if (bVar5 == null) {
            f2 = Integer.MIN_VALUE;
        } else {
            if (allNodes.f1777c) {
                allNodes.c();
            }
            while (true) {
                if (i4 >= allNodes.f1780f) {
                    break;
                } else if (allNodes.f1779e[i4] == bVar5) {
                    i3 = i4;
                    break;
                } else {
                    i4++;
                }
            }
            f2 = allNodes.f(i3);
        }
        return requestKeyboardFocusForVirtualView(f2);
    }

    private boolean performActionForChild(int i, int i2, Bundle bundle) {
        if (i2 != 1) {
            if (i2 != 2) {
                if (i2 != 64) {
                    if (i2 != 128) {
                        return onPerformActionForVirtualView(i, i2, bundle);
                    }
                    return clearAccessibilityFocus(i);
                }
                return requestAccessibilityFocus(i);
            }
            return clearKeyboardFocusForVirtualView(i);
        }
        return requestKeyboardFocusForVirtualView(i);
    }

    private boolean performActionForHost(int i, Bundle bundle) {
        View view = this.mHost;
        AtomicInteger atomicInteger = q.f2214a;
        return view.performAccessibilityAction(i, bundle);
    }

    private boolean requestAccessibilityFocus(int i) {
        int i2;
        if (this.mManager.isEnabled() && this.mManager.isTouchExplorationEnabled() && (i2 = this.mAccessibilityFocusedVirtualViewId) != i) {
            if (i2 != Integer.MIN_VALUE) {
                clearAccessibilityFocus(i2);
            }
            this.mAccessibilityFocusedVirtualViewId = i;
            this.mHost.invalidate();
            sendEventForVirtualView(i, Calib3d.CALIB_THIN_PRISM_MODEL);
            return true;
        }
        return false;
    }

    private void updateHoveredVirtualView(int i) {
        int i2 = this.mHoveredVirtualViewId;
        if (i2 == i) {
            return;
        }
        this.mHoveredVirtualViewId = i;
        sendEventForVirtualView(i, 128);
        sendEventForVirtualView(i2, 256);
    }

    public final boolean clearKeyboardFocusForVirtualView(int i) {
        if (this.mKeyboardFocusedVirtualViewId != i) {
            return false;
        }
        this.mKeyboardFocusedVirtualViewId = Integer.MIN_VALUE;
        onVirtualViewKeyboardFocusChanged(i, false);
        sendEventForVirtualView(i, 8);
        return true;
    }

    public final boolean dispatchHoverEvent(MotionEvent motionEvent) {
        if (this.mManager.isEnabled() && this.mManager.isTouchExplorationEnabled()) {
            int action = motionEvent.getAction();
            if (action != 7 && action != 9) {
                if (action == 10 && this.mHoveredVirtualViewId != Integer.MIN_VALUE) {
                    updateHoveredVirtualView(Integer.MIN_VALUE);
                    return true;
                }
                return false;
            }
            int virtualViewAt = getVirtualViewAt(motionEvent.getX(), motionEvent.getY());
            updateHoveredVirtualView(virtualViewAt);
            return virtualViewAt != Integer.MIN_VALUE;
        }
        return false;
    }

    public final boolean dispatchKeyEvent(KeyEvent keyEvent) {
        int i = 0;
        if (keyEvent.getAction() != 1) {
            int keyCode = keyEvent.getKeyCode();
            if (keyCode != 61) {
                if (keyCode != 66) {
                    switch (keyCode) {
                        case 19:
                        case 20:
                        case 21:
                        case 22:
                            if (keyEvent.hasNoModifiers()) {
                                int keyToDirection = keyToDirection(keyCode);
                                int repeatCount = keyEvent.getRepeatCount() + 1;
                                boolean z = false;
                                while (i < repeatCount && moveFocus(keyToDirection, null)) {
                                    i++;
                                    z = true;
                                }
                                return z;
                            }
                            return false;
                        case 23:
                            break;
                        default:
                            return false;
                    }
                }
                if (keyEvent.hasNoModifiers() && keyEvent.getRepeatCount() == 0) {
                    clickKeyboardFocusedVirtualView();
                    return true;
                }
                return false;
            } else if (keyEvent.hasNoModifiers()) {
                return moveFocus(2, null);
            } else {
                if (keyEvent.hasModifiers(1)) {
                    return moveFocus(1, null);
                }
                return false;
            }
        }
        return false;
    }

    public final int getAccessibilityFocusedVirtualViewId() {
        return this.mAccessibilityFocusedVirtualViewId;
    }

    @Override // b.j.j.a
    public b.j.j.x.c getAccessibilityNodeProvider(View view) {
        if (this.mNodeProvider == null) {
            this.mNodeProvider = new c();
        }
        return this.mNodeProvider;
    }

    @Deprecated
    public int getFocusedVirtualView() {
        return getAccessibilityFocusedVirtualViewId();
    }

    public final int getKeyboardFocusedVirtualViewId() {
        return this.mKeyboardFocusedVirtualViewId;
    }

    public abstract int getVirtualViewAt(float f2, float f3);

    public abstract void getVisibleVirtualViews(List<Integer> list);

    public final void invalidateRoot() {
        invalidateVirtualView(-1, 1);
    }

    public final void invalidateVirtualView(int i) {
        invalidateVirtualView(i, 0);
    }

    public b.j.j.x.b obtainAccessibilityNodeInfo(int i) {
        if (i == -1) {
            return createNodeForHost();
        }
        return createNodeForChild(i);
    }

    public final void onFocusChanged(boolean z, int i, Rect rect) {
        int i2 = this.mKeyboardFocusedVirtualViewId;
        if (i2 != Integer.MIN_VALUE) {
            clearKeyboardFocusForVirtualView(i2);
        }
        if (z) {
            moveFocus(i, rect);
        }
    }

    @Override // b.j.j.a
    public void onInitializeAccessibilityEvent(View view, AccessibilityEvent accessibilityEvent) {
        super.onInitializeAccessibilityEvent(view, accessibilityEvent);
        onPopulateEventForHost(accessibilityEvent);
    }

    @Override // b.j.j.a
    public void onInitializeAccessibilityNodeInfo(View view, b.j.j.x.b bVar) {
        super.onInitializeAccessibilityNodeInfo(view, bVar);
        onPopulateNodeForHost(bVar);
    }

    public abstract boolean onPerformActionForVirtualView(int i, int i2, Bundle bundle);

    public void onPopulateEventForHost(AccessibilityEvent accessibilityEvent) {
    }

    public void onPopulateEventForVirtualView(int i, AccessibilityEvent accessibilityEvent) {
    }

    public void onPopulateNodeForHost(b.j.j.x.b bVar) {
    }

    public abstract void onPopulateNodeForVirtualView(int i, b.j.j.x.b bVar);

    public void onVirtualViewKeyboardFocusChanged(int i, boolean z) {
    }

    public boolean performAction(int i, int i2, Bundle bundle) {
        if (i != -1) {
            return performActionForChild(i, i2, bundle);
        }
        return performActionForHost(i2, bundle);
    }

    public final boolean requestKeyboardFocusForVirtualView(int i) {
        int i2;
        if ((this.mHost.isFocused() || this.mHost.requestFocus()) && (i2 = this.mKeyboardFocusedVirtualViewId) != i) {
            if (i2 != Integer.MIN_VALUE) {
                clearKeyboardFocusForVirtualView(i2);
            }
            if (i == Integer.MIN_VALUE) {
                return false;
            }
            this.mKeyboardFocusedVirtualViewId = i;
            onVirtualViewKeyboardFocusChanged(i, true);
            sendEventForVirtualView(i, 8);
            return true;
        }
        return false;
    }

    public final boolean sendEventForVirtualView(int i, int i2) {
        ViewParent parent;
        if (i == Integer.MIN_VALUE || !this.mManager.isEnabled() || (parent = this.mHost.getParent()) == null) {
            return false;
        }
        return parent.requestSendAccessibilityEvent(this.mHost, createEvent(i, i2));
    }

    public final void invalidateVirtualView(int i, int i2) {
        ViewParent parent;
        if (i == Integer.MIN_VALUE || !this.mManager.isEnabled() || (parent = this.mHost.getParent()) == null) {
            return;
        }
        AccessibilityEvent createEvent = createEvent(i, 2048);
        createEvent.setContentChangeTypes(i2);
        parent.requestSendAccessibilityEvent(this.mHost, createEvent);
    }
}