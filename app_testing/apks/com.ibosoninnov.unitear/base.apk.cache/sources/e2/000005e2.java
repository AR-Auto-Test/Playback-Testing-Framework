package b.z;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.util.Property;
import android.view.View;
import android.view.ViewGroup;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: ChangeBounds.java */
/* loaded from: classes.dex */
public class b extends j {

    /* renamed from: b  reason: collision with root package name */
    public static final String[] f2850b = {"android:changeBounds:bounds", "android:changeBounds:clip", "android:changeBounds:parent", "android:changeBounds:windowX", "android:changeBounds:windowY"};

    /* renamed from: c  reason: collision with root package name */
    public static final Property<Drawable, PointF> f2851c = new a(PointF.class, "boundsOrigin");

    /* renamed from: d  reason: collision with root package name */
    public static final Property<i, PointF> f2852d = new C0056b(PointF.class, "topLeft");

    /* renamed from: e  reason: collision with root package name */
    public static final Property<i, PointF> f2853e = new c(PointF.class, "bottomRight");

    /* renamed from: f  reason: collision with root package name */
    public static final Property<View, PointF> f2854f = new d(PointF.class, "bottomRight");

    /* renamed from: g  reason: collision with root package name */
    public static final Property<View, PointF> f2855g = new e(PointF.class, "topLeft");

    /* renamed from: h  reason: collision with root package name */
    public static final Property<View, PointF> f2856h = new f(PointF.class, "position");
    public static b.z.g i = new b.z.g();
    public int[] j = new int[2];

    /* compiled from: ChangeBounds.java */
    /* loaded from: classes.dex */
    public static class a extends Property<Drawable, PointF> {

        /* renamed from: a  reason: collision with root package name */
        public Rect f2857a;

        public a(Class cls, String str) {
            super(cls, str);
            this.f2857a = new Rect();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.util.Property
        public PointF get(Drawable drawable) {
            drawable.copyBounds(this.f2857a);
            Rect rect = this.f2857a;
            return new PointF(rect.left, rect.top);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // android.util.Property
        public void set(Drawable drawable, PointF pointF) {
            Drawable drawable2 = drawable;
            PointF pointF2 = pointF;
            drawable2.copyBounds(this.f2857a);
            this.f2857a.offsetTo(Math.round(pointF2.x), Math.round(pointF2.y));
            drawable2.setBounds(this.f2857a);
        }
    }

    /* compiled from: ChangeBounds.java */
    /* renamed from: b.z.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0056b extends Property<i, PointF> {
        public C0056b(Class cls, String str) {
            super(cls, str);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.util.Property
        public /* bridge */ /* synthetic */ PointF get(i iVar) {
            return null;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // android.util.Property
        public void set(i iVar, PointF pointF) {
            i iVar2 = iVar;
            PointF pointF2 = pointF;
            Objects.requireNonNull(iVar2);
            iVar2.f2861a = Math.round(pointF2.x);
            int round = Math.round(pointF2.y);
            iVar2.f2862b = round;
            int i = iVar2.f2866f + 1;
            iVar2.f2866f = i;
            if (i == iVar2.f2867g) {
                s.b(iVar2.f2865e, iVar2.f2861a, round, iVar2.f2863c, iVar2.f2864d);
                iVar2.f2866f = 0;
                iVar2.f2867g = 0;
            }
        }
    }

    /* compiled from: ChangeBounds.java */
    /* loaded from: classes.dex */
    public static class c extends Property<i, PointF> {
        public c(Class cls, String str) {
            super(cls, str);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.util.Property
        public /* bridge */ /* synthetic */ PointF get(i iVar) {
            return null;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // android.util.Property
        public void set(i iVar, PointF pointF) {
            i iVar2 = iVar;
            PointF pointF2 = pointF;
            Objects.requireNonNull(iVar2);
            iVar2.f2863c = Math.round(pointF2.x);
            int round = Math.round(pointF2.y);
            iVar2.f2864d = round;
            int i = iVar2.f2867g + 1;
            iVar2.f2867g = i;
            if (iVar2.f2866f == i) {
                s.b(iVar2.f2865e, iVar2.f2861a, iVar2.f2862b, iVar2.f2863c, round);
                iVar2.f2866f = 0;
                iVar2.f2867g = 0;
            }
        }
    }

    /* compiled from: ChangeBounds.java */
    /* loaded from: classes.dex */
    public static class d extends Property<View, PointF> {
        public d(Class cls, String str) {
            super(cls, str);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.util.Property
        public /* bridge */ /* synthetic */ PointF get(View view) {
            return null;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // android.util.Property
        public void set(View view, PointF pointF) {
            View view2 = view;
            PointF pointF2 = pointF;
            s.b(view2, view2.getLeft(), view2.getTop(), Math.round(pointF2.x), Math.round(pointF2.y));
        }
    }

    /* compiled from: ChangeBounds.java */
    /* loaded from: classes.dex */
    public static class e extends Property<View, PointF> {
        public e(Class cls, String str) {
            super(cls, str);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.util.Property
        public /* bridge */ /* synthetic */ PointF get(View view) {
            return null;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // android.util.Property
        public void set(View view, PointF pointF) {
            View view2 = view;
            PointF pointF2 = pointF;
            s.b(view2, Math.round(pointF2.x), Math.round(pointF2.y), view2.getRight(), view2.getBottom());
        }
    }

    /* compiled from: ChangeBounds.java */
    /* loaded from: classes.dex */
    public static class f extends Property<View, PointF> {
        public f(Class cls, String str) {
            super(cls, str);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.util.Property
        public /* bridge */ /* synthetic */ PointF get(View view) {
            return null;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // android.util.Property
        public void set(View view, PointF pointF) {
            View view2 = view;
            PointF pointF2 = pointF;
            int round = Math.round(pointF2.x);
            int round2 = Math.round(pointF2.y);
            s.b(view2, round, round2, view2.getWidth() + round, view2.getHeight() + round2);
        }
    }

    /* compiled from: ChangeBounds.java */
    /* loaded from: classes.dex */
    public class g extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ i f2858a;
        private i mViewBounds;

        public g(b bVar, i iVar) {
            this.f2858a = iVar;
            this.mViewBounds = iVar;
        }
    }

    /* compiled from: ChangeBounds.java */
    /* loaded from: classes.dex */
    public class h extends k {

        /* renamed from: a  reason: collision with root package name */
        public boolean f2859a = false;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ ViewGroup f2860b;

        public h(b bVar, ViewGroup viewGroup) {
            this.f2860b = viewGroup;
        }

        @Override // b.z.k, b.z.j.f
        public void onTransitionCancel(j jVar) {
            r.a(this.f2860b, false);
            this.f2859a = true;
        }

        @Override // b.z.j.f
        public void onTransitionEnd(j jVar) {
            if (!this.f2859a) {
                r.a(this.f2860b, false);
            }
            jVar.removeListener(this);
        }

        @Override // b.z.k, b.z.j.f
        public void onTransitionPause(j jVar) {
            r.a(this.f2860b, false);
        }

        @Override // b.z.k, b.z.j.f
        public void onTransitionResume(j jVar) {
            r.a(this.f2860b, true);
        }
    }

    /* compiled from: ChangeBounds.java */
    /* loaded from: classes.dex */
    public static class i {

        /* renamed from: a  reason: collision with root package name */
        public int f2861a;

        /* renamed from: b  reason: collision with root package name */
        public int f2862b;

        /* renamed from: c  reason: collision with root package name */
        public int f2863c;

        /* renamed from: d  reason: collision with root package name */
        public int f2864d;

        /* renamed from: e  reason: collision with root package name */
        public View f2865e;

        /* renamed from: f  reason: collision with root package name */
        public int f2866f;

        /* renamed from: g  reason: collision with root package name */
        public int f2867g;

        public i(View view) {
            this.f2865e = view;
        }
    }

    @Override // b.z.j
    public void captureEndValues(p pVar) {
        captureValues(pVar);
    }

    @Override // b.z.j
    public void captureStartValues(p pVar) {
        captureValues(pVar);
    }

    public final void captureValues(p pVar) {
        View view = pVar.f2914b;
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        if (!view.isLaidOut() && view.getWidth() == 0 && view.getHeight() == 0) {
            return;
        }
        pVar.f2913a.put("android:changeBounds:bounds", new Rect(view.getLeft(), view.getTop(), view.getRight(), view.getBottom()));
        pVar.f2913a.put("android:changeBounds:parent", pVar.f2914b.getParent());
    }

    /* JADX DEBUG: Multi-variable search result rejected for r6v9, resolved type: android.animation.AnimatorSet */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // b.z.j
    public Animator createAnimator(ViewGroup viewGroup, p pVar, p pVar2) {
        int i2;
        b bVar;
        ObjectAnimator p;
        if (pVar == null || pVar2 == null) {
            return null;
        }
        Map<String, Object> map = pVar.f2913a;
        Map<String, Object> map2 = pVar2.f2913a;
        ViewGroup viewGroup2 = (ViewGroup) map.get("android:changeBounds:parent");
        ViewGroup viewGroup3 = (ViewGroup) map2.get("android:changeBounds:parent");
        if (viewGroup2 == null || viewGroup3 == null) {
            return null;
        }
        View view = pVar2.f2914b;
        Rect rect = (Rect) pVar.f2913a.get("android:changeBounds:bounds");
        Rect rect2 = (Rect) pVar2.f2913a.get("android:changeBounds:bounds");
        int i3 = rect.left;
        int i4 = rect2.left;
        int i5 = rect.top;
        int i6 = rect2.top;
        int i7 = rect.right;
        int i8 = rect2.right;
        int i9 = rect.bottom;
        int i10 = rect2.bottom;
        int i11 = i7 - i3;
        int i12 = i9 - i5;
        int i13 = i8 - i4;
        int i14 = i10 - i6;
        Rect rect3 = (Rect) pVar.f2913a.get("android:changeBounds:clip");
        Rect rect4 = (Rect) pVar2.f2913a.get("android:changeBounds:clip");
        if ((i11 == 0 || i12 == 0) && (i13 == 0 || i14 == 0)) {
            i2 = 0;
        } else {
            i2 = (i3 == i4 && i5 == i6) ? 0 : 1;
            if (i7 != i8 || i9 != i10) {
                i2++;
            }
        }
        if ((rect3 != null && !rect3.equals(rect4)) || (rect3 == null && rect4 != null)) {
            i2++;
        }
        int i15 = i2;
        if (i15 > 0) {
            s.b(view, i3, i5, i7, i9);
            if (i15 != 2) {
                bVar = this;
                if (i3 == i4 && i5 == i6) {
                    p = b.v.u.c.p(view, f2854f, getPathMotion().getPath(i7, i9, i8, i10));
                } else {
                    p = b.v.u.c.p(view, f2855g, getPathMotion().getPath(i3, i5, i4, i6));
                }
            } else if (i11 == i13 && i12 == i14) {
                p = b.v.u.c.p(view, f2856h, getPathMotion().getPath(i3, i5, i4, i6));
                bVar = this;
            } else {
                i iVar = new i(view);
                ObjectAnimator p2 = b.v.u.c.p(iVar, f2852d, getPathMotion().getPath(i3, i5, i4, i6));
                ObjectAnimator p3 = b.v.u.c.p(iVar, f2853e, getPathMotion().getPath(i7, i9, i8, i10));
                AnimatorSet animatorSet = new AnimatorSet();
                animatorSet.playTogether(p2, p3);
                bVar = this;
                animatorSet.addListener(new g(bVar, iVar));
                p = animatorSet;
            }
            if (view.getParent() instanceof ViewGroup) {
                ViewGroup viewGroup4 = (ViewGroup) view.getParent();
                r.a(viewGroup4, true);
                bVar.addListener(new h(bVar, viewGroup4));
            }
            return p;
        }
        return null;
    }

    @Override // b.z.j
    public String[] getTransitionProperties() {
        return f2850b;
    }
}