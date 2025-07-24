package b.j.j;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Rect;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.util.SparseArray;
import android.view.KeyEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.WindowInsets;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityManager;
import b.j.j.a;
import b.j.j.x.b;
import com.ibosoninnov.unitear.R;
import java.lang.ref.WeakReference;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: ViewCompat.java */
@SuppressLint({"PrivateConstructorForUtilityClass"})
/* loaded from: classes.dex */
public class q {

    /* renamed from: c  reason: collision with root package name */
    public static Field f2216c;

    /* renamed from: a  reason: collision with root package name */
    public static final AtomicInteger f2214a = new AtomicInteger(1);

    /* renamed from: b  reason: collision with root package name */
    public static WeakHashMap<View, s> f2215b = null;

    /* renamed from: d  reason: collision with root package name */
    public static boolean f2217d = false;

    /* renamed from: e  reason: collision with root package name */
    public static final int[] f2218e = {R.id.accessibility_custom_action_0, R.id.accessibility_custom_action_1, R.id.accessibility_custom_action_2, R.id.accessibility_custom_action_3, R.id.accessibility_custom_action_4, R.id.accessibility_custom_action_5, R.id.accessibility_custom_action_6, R.id.accessibility_custom_action_7, R.id.accessibility_custom_action_8, R.id.accessibility_custom_action_9, R.id.accessibility_custom_action_10, R.id.accessibility_custom_action_11, R.id.accessibility_custom_action_12, R.id.accessibility_custom_action_13, R.id.accessibility_custom_action_14, R.id.accessibility_custom_action_15, R.id.accessibility_custom_action_16, R.id.accessibility_custom_action_17, R.id.accessibility_custom_action_18, R.id.accessibility_custom_action_19, R.id.accessibility_custom_action_20, R.id.accessibility_custom_action_21, R.id.accessibility_custom_action_22, R.id.accessibility_custom_action_23, R.id.accessibility_custom_action_24, R.id.accessibility_custom_action_25, R.id.accessibility_custom_action_26, R.id.accessibility_custom_action_27, R.id.accessibility_custom_action_28, R.id.accessibility_custom_action_29, R.id.accessibility_custom_action_30, R.id.accessibility_custom_action_31};

    /* compiled from: ViewCompat.java */
    /* loaded from: classes.dex */
    public static class b {

        /* compiled from: ViewCompat.java */
        /* loaded from: classes.dex */
        public class a implements View.OnApplyWindowInsetsListener {

            /* renamed from: a  reason: collision with root package name */
            public w f2223a = null;

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ View f2224b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ j f2225c;

            public a(View view, j jVar) {
                this.f2224b = view;
                this.f2225c = jVar;
            }

            @Override // android.view.View.OnApplyWindowInsetsListener
            public WindowInsets onApplyWindowInsets(View view, WindowInsets windowInsets) {
                w k = w.k(windowInsets, view);
                int i = Build.VERSION.SDK_INT;
                if (i < 30) {
                    b.a(windowInsets, this.f2224b);
                    if (k.equals(this.f2223a)) {
                        return this.f2225c.onApplyWindowInsets(view, k).i();
                    }
                }
                this.f2223a = k;
                w onApplyWindowInsets = this.f2225c.onApplyWindowInsets(view, k);
                if (i >= 30) {
                    return onApplyWindowInsets.i();
                }
                AtomicInteger atomicInteger = q.f2214a;
                view.requestApplyInsets();
                return onApplyWindowInsets.i();
            }
        }

        public static void a(WindowInsets windowInsets, View view) {
            View.OnApplyWindowInsetsListener onApplyWindowInsetsListener = (View.OnApplyWindowInsetsListener) view.getTag(R.id.tag_window_insets_animation_callback);
            if (onApplyWindowInsetsListener != null) {
                onApplyWindowInsetsListener.onApplyWindowInsets(view, windowInsets);
            }
        }

        public static w b(View view, w wVar, Rect rect) {
            WindowInsets i = wVar.i();
            if (i != null) {
                return w.k(view.computeSystemWindowInsets(i, rect), view);
            }
            rect.setEmpty();
            return wVar;
        }

        public static void c(View view, j jVar) {
            if (Build.VERSION.SDK_INT < 30) {
                view.setTag(R.id.tag_on_apply_window_listener, jVar);
            }
            if (jVar == null) {
                view.setOnApplyWindowInsetsListener((View.OnApplyWindowInsetsListener) view.getTag(R.id.tag_window_insets_animation_callback));
            } else {
                view.setOnApplyWindowInsetsListener(new a(view, jVar));
            }
        }
    }

    /* compiled from: ViewCompat.java */
    /* loaded from: classes.dex */
    public static class c {
        public static w a(View view) {
            WindowInsets rootWindowInsets = view.getRootWindowInsets();
            if (rootWindowInsets == null) {
                return null;
            }
            w k = w.k(rootWindowInsets, null);
            k.f2238b.n(k);
            k.f2238b.d(view.getRootView());
            return k;
        }
    }

    /* compiled from: ViewCompat.java */
    /* loaded from: classes.dex */
    public static class d {
        public static void a(View view, Context context, int[] iArr, AttributeSet attributeSet, TypedArray typedArray, int i, int i2) {
            view.saveAttributeDataForStyleable(context, iArr, attributeSet, typedArray, i, i2);
        }
    }

    /* compiled from: ViewCompat.java */
    /* loaded from: classes.dex */
    public interface e {
        boolean a(View view, KeyEvent keyEvent);
    }

    /* compiled from: ViewCompat.java */
    /* loaded from: classes.dex */
    public static class f {

        /* renamed from: a  reason: collision with root package name */
        public static final ArrayList<WeakReference<View>> f2226a = new ArrayList<>();

        /* renamed from: b  reason: collision with root package name */
        public WeakHashMap<View, Boolean> f2227b = null;

        /* renamed from: c  reason: collision with root package name */
        public SparseArray<WeakReference<View>> f2228c = null;

        /* renamed from: d  reason: collision with root package name */
        public WeakReference<KeyEvent> f2229d = null;

        public final View a(View view, KeyEvent keyEvent) {
            WeakHashMap<View, Boolean> weakHashMap = this.f2227b;
            if (weakHashMap != null && weakHashMap.containsKey(view)) {
                if (view instanceof ViewGroup) {
                    ViewGroup viewGroup = (ViewGroup) view;
                    for (int childCount = viewGroup.getChildCount() - 1; childCount >= 0; childCount--) {
                        View a2 = a(viewGroup.getChildAt(childCount), keyEvent);
                        if (a2 != null) {
                            return a2;
                        }
                    }
                }
                if (b(view, keyEvent)) {
                    return view;
                }
            }
            return null;
        }

        public final boolean b(View view, KeyEvent keyEvent) {
            ArrayList arrayList = (ArrayList) view.getTag(R.id.tag_unhandled_key_listeners);
            if (arrayList != null) {
                for (int size = arrayList.size() - 1; size >= 0; size--) {
                    if (((e) arrayList.get(size)).a(view, keyEvent)) {
                        return true;
                    }
                }
                return false;
            }
            return false;
        }
    }

    static {
        new WeakHashMap();
    }

    public static void a(View view, b.a aVar) {
        b.j.j.a e2 = e(view);
        if (e2 == null) {
            e2 = new b.j.j.a();
        }
        n(view, e2);
        k(aVar.a(), view);
        h(view).add(aVar);
        i(view, 0);
    }

    public static s b(View view) {
        if (f2215b == null) {
            f2215b = new WeakHashMap<>();
        }
        s sVar = f2215b.get(view);
        if (sVar == null) {
            s sVar2 = new s(view);
            f2215b.put(view, sVar2);
            return sVar2;
        }
        return sVar;
    }

    public static w c(View view, w wVar) {
        WindowInsets i = wVar.i();
        if (i != null) {
            WindowInsets dispatchApplyWindowInsets = view.dispatchApplyWindowInsets(i);
            if (!dispatchApplyWindowInsets.equals(i)) {
                return w.k(dispatchApplyWindowInsets, view);
            }
        }
        return wVar;
    }

    public static boolean d(View view, KeyEvent keyEvent) {
        if (Build.VERSION.SDK_INT >= 28) {
            return false;
        }
        ArrayList<WeakReference<View>> arrayList = f.f2226a;
        f fVar = (f) view.getTag(R.id.tag_unhandled_key_event_manager);
        if (fVar == null) {
            fVar = new f();
            view.setTag(R.id.tag_unhandled_key_event_manager, fVar);
        }
        if (keyEvent.getAction() == 0) {
            WeakHashMap<View, Boolean> weakHashMap = fVar.f2227b;
            if (weakHashMap != null) {
                weakHashMap.clear();
            }
            ArrayList<WeakReference<View>> arrayList2 = f.f2226a;
            if (!arrayList2.isEmpty()) {
                synchronized (arrayList2) {
                    if (fVar.f2227b == null) {
                        fVar.f2227b = new WeakHashMap<>();
                    }
                    int size = arrayList2.size();
                    while (true) {
                        size--;
                        if (size < 0) {
                            break;
                        }
                        ArrayList<WeakReference<View>> arrayList3 = f.f2226a;
                        View view2 = arrayList3.get(size).get();
                        if (view2 == null) {
                            arrayList3.remove(size);
                        } else {
                            fVar.f2227b.put(view2, Boolean.TRUE);
                            for (ViewParent parent = view2.getParent(); parent instanceof View; parent = parent.getParent()) {
                                fVar.f2227b.put((View) parent, Boolean.TRUE);
                            }
                        }
                    }
                }
            }
        }
        View a2 = fVar.a(view, keyEvent);
        if (keyEvent.getAction() == 0) {
            int keyCode = keyEvent.getKeyCode();
            if (a2 != null && !KeyEvent.isModifierKey(keyCode)) {
                if (fVar.f2228c == null) {
                    fVar.f2228c = new SparseArray<>();
                }
                fVar.f2228c.put(keyCode, new WeakReference<>(a2));
            }
        }
        return a2 != null;
    }

    public static b.j.j.a e(View view) {
        View.AccessibilityDelegate f2 = f(view);
        if (f2 == null) {
            return null;
        }
        if (f2 instanceof a.C0036a) {
            return ((a.C0036a) f2).f2197a;
        }
        return new b.j.j.a(f2);
    }

    public static View.AccessibilityDelegate f(View view) {
        if (Build.VERSION.SDK_INT >= 29) {
            return view.getAccessibilityDelegate();
        }
        if (f2217d) {
            return null;
        }
        if (f2216c == null) {
            try {
                Field declaredField = View.class.getDeclaredField("mAccessibilityDelegate");
                f2216c = declaredField;
                declaredField.setAccessible(true);
            } catch (Throwable unused) {
                f2217d = true;
                return null;
            }
        }
        try {
            Object obj = f2216c.get(view);
            if (obj instanceof View.AccessibilityDelegate) {
                return (View.AccessibilityDelegate) obj;
            }
            return null;
        } catch (Throwable unused2) {
            f2217d = true;
            return null;
        }
    }

    public static CharSequence g(View view) {
        return new n(R.id.tag_accessibility_pane_title, CharSequence.class, 8, 28).c(view);
    }

    public static List<b.a> h(View view) {
        ArrayList arrayList = (ArrayList) view.getTag(R.id.tag_accessibility_actions);
        if (arrayList == null) {
            ArrayList arrayList2 = new ArrayList();
            view.setTag(R.id.tag_accessibility_actions, arrayList2);
            return arrayList2;
        }
        return arrayList;
    }

    public static void i(View view, int i) {
        AccessibilityManager accessibilityManager = (AccessibilityManager) view.getContext().getSystemService("accessibility");
        if (accessibilityManager.isEnabled()) {
            boolean z = g(view) != null && view.getVisibility() == 0;
            if (view.getAccessibilityLiveRegion() != 0 || z) {
                AccessibilityEvent obtain = AccessibilityEvent.obtain();
                obtain.setEventType(z ? 32 : 2048);
                obtain.setContentChangeTypes(i);
                if (z) {
                    obtain.getText().add(g(view));
                    if (view.getImportantForAccessibility() == 0) {
                        view.setImportantForAccessibility(1);
                    }
                    ViewParent parent = view.getParent();
                    while (true) {
                        if (!(parent instanceof View)) {
                            break;
                        } else if (((View) parent).getImportantForAccessibility() == 4) {
                            view.setImportantForAccessibility(2);
                            break;
                        } else {
                            parent = parent.getParent();
                        }
                    }
                }
                view.sendAccessibilityEventUnchecked(obtain);
            } else if (i == 32) {
                AccessibilityEvent obtain2 = AccessibilityEvent.obtain();
                view.onInitializeAccessibilityEvent(obtain2);
                obtain2.setEventType(32);
                obtain2.setContentChangeTypes(i);
                obtain2.setSource(view);
                view.onPopulateAccessibilityEvent(obtain2);
                obtain2.getText().add(g(view));
                accessibilityManager.sendAccessibilityEvent(obtain2);
            } else if (view.getParent() != null) {
                try {
                    view.getParent().notifySubtreeAccessibilityStateChanged(view, view, i);
                } catch (AbstractMethodError e2) {
                    Log.e("ViewCompat", view.getParent().getClass().getSimpleName() + " does not fully implement ViewParent", e2);
                }
            }
        }
    }

    public static w j(View view, w wVar) {
        WindowInsets i = wVar.i();
        if (i != null) {
            WindowInsets onApplyWindowInsets = view.onApplyWindowInsets(i);
            if (!onApplyWindowInsets.equals(i)) {
                return w.k(onApplyWindowInsets, view);
            }
        }
        return wVar;
    }

    public static void k(int i, View view) {
        List<b.a> h2 = h(view);
        for (int i2 = 0; i2 < h2.size(); i2++) {
            if (h2.get(i2).a() == i) {
                h2.remove(i2);
                return;
            }
        }
    }

    public static void l(View view, b.a aVar, CharSequence charSequence, b.j.j.x.d dVar) {
        if (dVar == null) {
            k(aVar.a(), view);
            i(view, 0);
            return;
        }
        a(view, new b.a(null, aVar.m, null, dVar, aVar.n));
    }

    public static void m(View view, @SuppressLint({"ContextFirst"}) Context context, int[] iArr, AttributeSet attributeSet, TypedArray typedArray, int i, int i2) {
        if (Build.VERSION.SDK_INT >= 29) {
            d.a(view, context, iArr, attributeSet, typedArray, i, i2);
        }
    }

    public static void n(View view, b.j.j.a aVar) {
        if (aVar == null && (f(view) instanceof a.C0036a)) {
            aVar = new b.j.j.a();
        }
        view.setAccessibilityDelegate(aVar == null ? null : aVar.getBridge());
    }

    public static void o(View view, l lVar) {
        view.setPointerIcon(null);
    }

    /* compiled from: ViewCompat.java */
    /* loaded from: classes.dex */
    public static abstract class a<T> {

        /* renamed from: a  reason: collision with root package name */
        public final int f2219a;

        /* renamed from: b  reason: collision with root package name */
        public final Class<T> f2220b;

        /* renamed from: c  reason: collision with root package name */
        public final int f2221c;

        /* renamed from: d  reason: collision with root package name */
        public final int f2222d;

        public a(int i, Class<T> cls, int i2) {
            this.f2219a = i;
            this.f2220b = cls;
            this.f2222d = 0;
            this.f2221c = i2;
        }

        public boolean a(Boolean bool, Boolean bool2) {
            return (bool == null ? false : bool.booleanValue()) == (bool2 == null ? false : bool2.booleanValue());
        }

        public abstract T b(View view);

        public T c(View view) {
            if (Build.VERSION.SDK_INT >= this.f2221c) {
                return b(view);
            }
            T t = (T) view.getTag(this.f2219a);
            if (this.f2220b.isInstance(t)) {
                return t;
            }
            return null;
        }

        public a(int i, Class<T> cls, int i2, int i3) {
            this.f2219a = i;
            this.f2220b = cls;
            this.f2222d = i2;
            this.f2221c = i3;
        }
    }
}