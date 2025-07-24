package b.j.j.x;

import android.graphics.Rect;
import android.os.Build;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.TextUtils;
import android.text.style.ClickableSpan;
import android.view.View;
import android.view.accessibility.AccessibilityNodeInfo;
import b.j.j.x.d;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.opencv.calib3d.Calib3d;

/* compiled from: AccessibilityNodeInfoCompat.java */
/* loaded from: classes.dex */
public class b {

    /* renamed from: a  reason: collision with root package name */
    public static int f2258a;

    /* renamed from: b  reason: collision with root package name */
    public final AccessibilityNodeInfo f2259b;

    /* renamed from: c  reason: collision with root package name */
    public int f2260c = -1;

    /* renamed from: d  reason: collision with root package name */
    public int f2261d = -1;

    /* compiled from: AccessibilityNodeInfoCompat.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public static final a f2262a = new a(1, null);

        /* renamed from: b  reason: collision with root package name */
        public static final a f2263b = new a(2, null);

        /* renamed from: c  reason: collision with root package name */
        public static final a f2264c;

        /* renamed from: d  reason: collision with root package name */
        public static final a f2265d;

        /* renamed from: e  reason: collision with root package name */
        public static final a f2266e;

        /* renamed from: f  reason: collision with root package name */
        public static final a f2267f;

        /* renamed from: g  reason: collision with root package name */
        public static final a f2268g;

        /* renamed from: h  reason: collision with root package name */
        public static final a f2269h;
        public static final a i;
        public static final a j;
        public static final a k;
        public final Object l;
        public final int m;
        public final Class<? extends d.a> n;
        public final d o;

        static {
            new AccessibilityNodeInfo.AccessibilityAction(4, null);
            new AccessibilityNodeInfo.AccessibilityAction(8, null);
            f2264c = new a(16, null);
            new AccessibilityNodeInfo.AccessibilityAction(32, null);
            new AccessibilityNodeInfo.AccessibilityAction(64, null);
            new AccessibilityNodeInfo.AccessibilityAction(128, null);
            new AccessibilityNodeInfo.AccessibilityAction(256, null);
            new AccessibilityNodeInfo.AccessibilityAction(512, null);
            new AccessibilityNodeInfo.AccessibilityAction(1024, null);
            new AccessibilityNodeInfo.AccessibilityAction(2048, null);
            f2265d = new a(4096, null);
            f2266e = new a(8192, null);
            new AccessibilityNodeInfo.AccessibilityAction(Calib3d.CALIB_RATIONAL_MODEL, null);
            new AccessibilityNodeInfo.AccessibilityAction(Calib3d.CALIB_THIN_PRISM_MODEL, null);
            new AccessibilityNodeInfo.AccessibilityAction(65536, null);
            new AccessibilityNodeInfo.AccessibilityAction(131072, null);
            f2267f = new a(Calib3d.CALIB_TILTED_MODEL, null);
            f2268g = new a(524288, null);
            f2269h = new a(1048576, null);
            new AccessibilityNodeInfo.AccessibilityAction(Calib3d.CALIB_FIX_TANGENT_DIST, null);
            int i2 = Build.VERSION.SDK_INT;
            if (AccessibilityNodeInfo.AccessibilityAction.ACTION_SHOW_ON_SCREEN == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908342, null);
            }
            if (AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_TO_POSITION == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908343, null);
            }
            i = new a(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_UP, 16908344, null, null, null);
            if (AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_LEFT == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908345, null);
            }
            j = new a(AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_DOWN, 16908346, null, null, null);
            if (AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_RIGHT == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908347, null);
            }
            if ((i2 >= 29 ? AccessibilityNodeInfo.AccessibilityAction.ACTION_PAGE_UP : null) == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908358, null);
            }
            if ((i2 >= 29 ? AccessibilityNodeInfo.AccessibilityAction.ACTION_PAGE_DOWN : null) == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908359, null);
            }
            if ((i2 >= 29 ? AccessibilityNodeInfo.AccessibilityAction.ACTION_PAGE_LEFT : null) == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908360, null);
            }
            if ((i2 >= 29 ? AccessibilityNodeInfo.AccessibilityAction.ACTION_PAGE_RIGHT : null) == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908361, null);
            }
            if (AccessibilityNodeInfo.AccessibilityAction.ACTION_CONTEXT_CLICK == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908348, null);
            }
            k = new a(AccessibilityNodeInfo.AccessibilityAction.ACTION_SET_PROGRESS, 16908349, null, null, d.b.class);
            if ((i2 >= 26 ? AccessibilityNodeInfo.AccessibilityAction.ACTION_MOVE_WINDOW : null) == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908354, null);
            }
            if ((i2 >= 28 ? AccessibilityNodeInfo.AccessibilityAction.ACTION_SHOW_TOOLTIP : null) == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908356, null);
            }
            if ((i2 >= 28 ? AccessibilityNodeInfo.AccessibilityAction.ACTION_HIDE_TOOLTIP : null) == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908357, null);
            }
            if ((i2 >= 30 ? AccessibilityNodeInfo.AccessibilityAction.ACTION_PRESS_AND_HOLD : null) == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908362, null);
            }
            if ((i2 >= 30 ? AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER : null) == null) {
                new AccessibilityNodeInfo.AccessibilityAction(16908372, null);
            }
        }

        public a(int i2, CharSequence charSequence) {
            this(null, i2, charSequence, null, null);
        }

        public int a() {
            return ((AccessibilityNodeInfo.AccessibilityAction) this.l).getId();
        }

        public boolean equals(Object obj) {
            if (obj != null && (obj instanceof a)) {
                a aVar = (a) obj;
                Object obj2 = this.l;
                return obj2 == null ? aVar.l == null : obj2.equals(aVar.l);
            }
            return false;
        }

        public int hashCode() {
            Object obj = this.l;
            if (obj != null) {
                return obj.hashCode();
            }
            return 0;
        }

        public a(Object obj, int i2, CharSequence charSequence, d dVar, Class<? extends d.a> cls) {
            this.m = i2;
            this.o = dVar;
            if (obj == null) {
                this.l = new AccessibilityNodeInfo.AccessibilityAction(i2, charSequence);
            } else {
                this.l = obj;
            }
            this.n = cls;
        }
    }

    /* compiled from: AccessibilityNodeInfoCompat.java */
    /* renamed from: b.j.j.x.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0037b {

        /* renamed from: a  reason: collision with root package name */
        public final Object f2270a;

        public C0037b(Object obj) {
            this.f2270a = obj;
        }

        public static C0037b a(int i, int i2, boolean z, int i3) {
            return new C0037b(AccessibilityNodeInfo.CollectionInfo.obtain(i, i2, z, i3));
        }
    }

    /* compiled from: AccessibilityNodeInfoCompat.java */
    /* loaded from: classes.dex */
    public static class c {

        /* renamed from: a  reason: collision with root package name */
        public final Object f2271a;

        public c(Object obj) {
            this.f2271a = obj;
        }

        public static c a(int i, int i2, int i3, int i4, boolean z, boolean z2) {
            return new c(AccessibilityNodeInfo.CollectionItemInfo.obtain(i, i2, i3, i4, z, z2));
        }
    }

    public b(AccessibilityNodeInfo accessibilityNodeInfo) {
        this.f2259b = accessibilityNodeInfo;
    }

    public static String c(int i) {
        if (i != 1) {
            if (i != 2) {
                switch (i) {
                    case 4:
                        return "ACTION_SELECT";
                    case 8:
                        return "ACTION_CLEAR_SELECTION";
                    case 16:
                        return "ACTION_CLICK";
                    case 32:
                        return "ACTION_LONG_CLICK";
                    case 64:
                        return "ACTION_ACCESSIBILITY_FOCUS";
                    case 128:
                        return "ACTION_CLEAR_ACCESSIBILITY_FOCUS";
                    case 256:
                        return "ACTION_NEXT_AT_MOVEMENT_GRANULARITY";
                    case 512:
                        return "ACTION_PREVIOUS_AT_MOVEMENT_GRANULARITY";
                    case 1024:
                        return "ACTION_NEXT_HTML_ELEMENT";
                    case 2048:
                        return "ACTION_PREVIOUS_HTML_ELEMENT";
                    case 4096:
                        return "ACTION_SCROLL_FORWARD";
                    case 8192:
                        return "ACTION_SCROLL_BACKWARD";
                    case Calib3d.CALIB_RATIONAL_MODEL /* 16384 */:
                        return "ACTION_COPY";
                    case Calib3d.CALIB_THIN_PRISM_MODEL /* 32768 */:
                        return "ACTION_PASTE";
                    case 65536:
                        return "ACTION_CUT";
                    case 131072:
                        return "ACTION_SET_SELECTION";
                    case Calib3d.CALIB_TILTED_MODEL /* 262144 */:
                        return "ACTION_EXPAND";
                    case 524288:
                        return "ACTION_COLLAPSE";
                    case Calib3d.CALIB_FIX_TANGENT_DIST /* 2097152 */:
                        return "ACTION_SET_TEXT";
                    case 16908354:
                        return "ACTION_MOVE_WINDOW";
                    case 16908372:
                        return "ACTION_IME_ENTER";
                    default:
                        switch (i) {
                            case 16908342:
                                return "ACTION_SHOW_ON_SCREEN";
                            case 16908343:
                                return "ACTION_SCROLL_TO_POSITION";
                            case 16908344:
                                return "ACTION_SCROLL_UP";
                            case 16908345:
                                return "ACTION_SCROLL_LEFT";
                            case 16908346:
                                return "ACTION_SCROLL_DOWN";
                            case 16908347:
                                return "ACTION_SCROLL_RIGHT";
                            case 16908348:
                                return "ACTION_CONTEXT_CLICK";
                            case 16908349:
                                return "ACTION_SET_PROGRESS";
                            default:
                                switch (i) {
                                    case 16908356:
                                        return "ACTION_SHOW_TOOLTIP";
                                    case 16908357:
                                        return "ACTION_HIDE_TOOLTIP";
                                    case 16908358:
                                        return "ACTION_PAGE_UP";
                                    case 16908359:
                                        return "ACTION_PAGE_DOWN";
                                    case 16908360:
                                        return "ACTION_PAGE_LEFT";
                                    case 16908361:
                                        return "ACTION_PAGE_RIGHT";
                                    case 16908362:
                                        return "ACTION_PRESS_AND_HOLD";
                                    default:
                                        return "ACTION_UNKNOWN";
                                }
                        }
                }
            }
            return "ACTION_CLEAR_FOCUS";
        }
        return "ACTION_FOCUS";
    }

    public static ClickableSpan[] f(CharSequence charSequence) {
        if (charSequence instanceof Spanned) {
            return (ClickableSpan[]) ((Spanned) charSequence).getSpans(0, charSequence.length(), ClickableSpan.class);
        }
        return null;
    }

    public void a(a aVar) {
        this.f2259b.addAction((AccessibilityNodeInfo.AccessibilityAction) aVar.l);
    }

    public final List<Integer> b(String str) {
        ArrayList<Integer> integerArrayList = this.f2259b.getExtras().getIntegerArrayList(str);
        if (integerArrayList == null) {
            ArrayList<Integer> arrayList = new ArrayList<>();
            this.f2259b.getExtras().putIntegerArrayList(str, arrayList);
            return arrayList;
        }
        return integerArrayList;
    }

    public int d() {
        return this.f2259b.getActions();
    }

    public CharSequence e() {
        return this.f2259b.getClassName();
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj != null && (obj instanceof b)) {
            b bVar = (b) obj;
            AccessibilityNodeInfo accessibilityNodeInfo = this.f2259b;
            if (accessibilityNodeInfo == null) {
                if (bVar.f2259b != null) {
                    return false;
                }
            } else if (!accessibilityNodeInfo.equals(bVar.f2259b)) {
                return false;
            }
            return this.f2261d == bVar.f2261d && this.f2260c == bVar.f2260c;
        }
        return false;
    }

    public CharSequence g() {
        return this.f2259b.getContentDescription();
    }

    public Bundle h() {
        return this.f2259b.getExtras();
    }

    public int hashCode() {
        AccessibilityNodeInfo accessibilityNodeInfo = this.f2259b;
        if (accessibilityNodeInfo == null) {
            return 0;
        }
        return accessibilityNodeInfo.hashCode();
    }

    public CharSequence i() {
        if (!b("androidx.view.accessibility.AccessibilityNodeInfoCompat.SPANS_START_KEY").isEmpty()) {
            List<Integer> b2 = b("androidx.view.accessibility.AccessibilityNodeInfoCompat.SPANS_START_KEY");
            List<Integer> b3 = b("androidx.view.accessibility.AccessibilityNodeInfoCompat.SPANS_END_KEY");
            List<Integer> b4 = b("androidx.view.accessibility.AccessibilityNodeInfoCompat.SPANS_FLAGS_KEY");
            List<Integer> b5 = b("androidx.view.accessibility.AccessibilityNodeInfoCompat.SPANS_ID_KEY");
            SpannableString spannableString = new SpannableString(TextUtils.substring(this.f2259b.getText(), 0, this.f2259b.getText().length()));
            for (int i = 0; i < b2.size(); i++) {
                spannableString.setSpan(new b.j.j.x.a(b5.get(i).intValue(), this, h().getInt("androidx.view.accessibility.AccessibilityNodeInfoCompat.SPANS_ACTION_ID_KEY")), b2.get(i).intValue(), b3.get(i).intValue(), b4.get(i).intValue());
            }
            return spannableString;
        }
        return this.f2259b.getText();
    }

    public boolean j() {
        return this.f2259b.isEnabled();
    }

    public boolean k(a aVar) {
        return this.f2259b.removeAction((AccessibilityNodeInfo.AccessibilityAction) aVar.l);
    }

    public final void l(int i, boolean z) {
        Bundle h2 = h();
        if (h2 != null) {
            int i2 = h2.getInt("androidx.view.accessibility.AccessibilityNodeInfoCompat.BOOLEAN_PROPERTY_KEY", 0) & (~i);
            if (!z) {
                i = 0;
            }
            h2.putInt("androidx.view.accessibility.AccessibilityNodeInfoCompat.BOOLEAN_PROPERTY_KEY", i | i2);
        }
    }

    public void m(Object obj) {
        this.f2259b.setCollectionInfo(obj == null ? null : (AccessibilityNodeInfo.CollectionInfo) ((C0037b) obj).f2270a);
    }

    public void n(Object obj) {
        this.f2259b.setCollectionItemInfo((AccessibilityNodeInfo.CollectionItemInfo) ((c) obj).f2271a);
    }

    public void o(CharSequence charSequence) {
        if (Build.VERSION.SDK_INT >= 26) {
            this.f2259b.setHintText(charSequence);
        } else {
            this.f2259b.getExtras().putCharSequence("androidx.view.accessibility.AccessibilityNodeInfoCompat.HINT_TEXT_KEY", charSequence);
        }
    }

    public void p(View view) {
        this.f2260c = -1;
        this.f2259b.setParent(view);
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:21:0x014b */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r3v2, types: [java.util.List] */
    /* JADX WARN: Type inference failed for: r3v3, types: [java.util.List] */
    /* JADX WARN: Type inference failed for: r3v4, types: [java.util.ArrayList] */
    public String toString() {
        ?? emptyList;
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        Rect rect = new Rect();
        this.f2259b.getBoundsInParent(rect);
        sb.append("; boundsInParent: " + rect);
        this.f2259b.getBoundsInScreen(rect);
        sb.append("; boundsInScreen: " + rect);
        sb.append("; packageName: ");
        sb.append(this.f2259b.getPackageName());
        sb.append("; className: ");
        sb.append(e());
        sb.append("; text: ");
        sb.append(i());
        sb.append("; contentDescription: ");
        sb.append(g());
        sb.append("; viewId: ");
        sb.append(this.f2259b.getViewIdResourceName());
        sb.append("; checkable: ");
        sb.append(this.f2259b.isCheckable());
        sb.append("; checked: ");
        sb.append(this.f2259b.isChecked());
        sb.append("; focusable: ");
        sb.append(this.f2259b.isFocusable());
        sb.append("; focused: ");
        sb.append(this.f2259b.isFocused());
        sb.append("; selected: ");
        sb.append(this.f2259b.isSelected());
        sb.append("; clickable: ");
        sb.append(this.f2259b.isClickable());
        sb.append("; longClickable: ");
        sb.append(this.f2259b.isLongClickable());
        sb.append("; enabled: ");
        sb.append(j());
        sb.append("; password: ");
        sb.append(this.f2259b.isPassword());
        sb.append("; scrollable: " + this.f2259b.isScrollable());
        sb.append("; [");
        List<AccessibilityNodeInfo.AccessibilityAction> actionList = this.f2259b.getActionList();
        if (actionList != null) {
            emptyList = new ArrayList();
            int size = actionList.size();
            for (int i = 0; i < size; i++) {
                emptyList.add(new a(actionList.get(i), 0, null, null, null));
            }
        } else {
            emptyList = Collections.emptyList();
        }
        for (int i2 = 0; i2 < emptyList.size(); i2++) {
            a aVar = (a) emptyList.get(i2);
            String c2 = c(aVar.a());
            if (c2.equals("ACTION_UNKNOWN") && ((AccessibilityNodeInfo.AccessibilityAction) aVar.l).getLabel() != null) {
                c2 = ((AccessibilityNodeInfo.AccessibilityAction) aVar.l).getLabel().toString();
            }
            sb.append(c2);
            if (i2 != emptyList.size() - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}