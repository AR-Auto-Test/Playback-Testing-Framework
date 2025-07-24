package c.a.a;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.drawable.Animatable;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.widget.ImageView;
import c.a.a.b0.h0.c;
import c.a.a.z.l.e;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/* compiled from: LottieDrawable.java */
/* loaded from: classes.dex */
public class j extends Drawable implements Drawable.Callback, Animatable {

    /* renamed from: b  reason: collision with root package name */
    public final Matrix f3074b = new Matrix();

    /* renamed from: c  reason: collision with root package name */
    public c.a.a.d f3075c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.c0.d f3076d;

    /* renamed from: e  reason: collision with root package name */
    public float f3077e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f3078f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f3079g;

    /* renamed from: h  reason: collision with root package name */
    public final ArrayList<o> f3080h;
    public final ValueAnimator.AnimatorUpdateListener i;
    public ImageView.ScaleType j;
    public c.a.a.y.b k;
    public String l;
    public c.a.a.b m;
    public c.a.a.y.a n;
    public boolean o;
    public c.a.a.z.l.c p;
    public int q;
    public boolean r;
    public boolean s;
    public boolean t;
    public boolean u;

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class a implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ String f3081a;

        public a(String str) {
            this.f3081a = str;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.q(this.f3081a);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class b implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f3083a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ int f3084b;

        public b(int i, int i2) {
            this.f3083a = i;
            this.f3084b = i2;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.p(this.f3083a, this.f3084b);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class c implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f3086a;

        public c(int i) {
            this.f3086a = i;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.l(this.f3086a);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class d implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ float f3088a;

        public d(float f2) {
            this.f3088a = f2;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.u(this.f3088a);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class e implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ c.a.a.z.e f3090a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Object f3091b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ c.a.a.d0.c f3092c;

        public e(c.a.a.z.e eVar, Object obj, c.a.a.d0.c cVar) {
            this.f3090a = eVar;
            this.f3091b = obj;
            this.f3092c = cVar;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.a(this.f3090a, this.f3091b, this.f3092c);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class f implements ValueAnimator.AnimatorUpdateListener {
        public f() {
        }

        @Override // android.animation.ValueAnimator.AnimatorUpdateListener
        public void onAnimationUpdate(ValueAnimator valueAnimator) {
            j jVar = j.this;
            c.a.a.z.l.c cVar = jVar.p;
            if (cVar != null) {
                cVar.p(jVar.f3076d.d());
            }
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class g implements o {
        public g() {
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.j();
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class h implements o {
        public h() {
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.k();
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class i implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f3097a;

        public i(int i) {
            this.f3097a = i;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.r(this.f3097a);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* renamed from: c.a.a.j$j  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0059j implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ float f3099a;

        public C0059j(float f2) {
            this.f3099a = f2;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.t(this.f3099a);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class k implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f3101a;

        public k(int i) {
            this.f3101a = i;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.m(this.f3101a);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class l implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ float f3103a;

        public l(float f2) {
            this.f3103a = f2;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.o(this.f3103a);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class m implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ String f3105a;

        public m(String str) {
            this.f3105a = str;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.s(this.f3105a);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public class n implements o {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ String f3107a;

        public n(String str) {
            this.f3107a = str;
        }

        @Override // c.a.a.j.o
        public void a(c.a.a.d dVar) {
            j.this.n(this.f3107a);
        }
    }

    /* compiled from: LottieDrawable.java */
    /* loaded from: classes.dex */
    public interface o {
        void a(c.a.a.d dVar);
    }

    public j() {
        c.a.a.c0.d dVar = new c.a.a.c0.d();
        this.f3076d = dVar;
        this.f3077e = 1.0f;
        this.f3078f = true;
        this.f3079g = false;
        new HashSet();
        this.f3080h = new ArrayList<>();
        f fVar = new f();
        this.i = fVar;
        this.q = 255;
        this.t = true;
        this.u = false;
        dVar.f3019b.add(fVar);
    }

    public <T> void a(c.a.a.z.e eVar, T t, c.a.a.d0.c<T> cVar) {
        List list;
        c.a.a.z.l.c cVar2 = this.p;
        if (cVar2 == null) {
            this.f3080h.add(new e(eVar, t, cVar));
            return;
        }
        c.a.a.z.f fVar = eVar.f3278b;
        boolean z = true;
        if (fVar != null) {
            fVar.h(t, cVar);
        } else {
            if (cVar2 == null) {
                c.a.a.c0.c.b("Cannot resolve KeyPath. Composition is not set yet.");
                list = Collections.emptyList();
            } else {
                ArrayList arrayList = new ArrayList();
                this.p.c(eVar, 0, arrayList, new c.a.a.z.e(new String[0]));
                list = arrayList;
            }
            for (int i2 = 0; i2 < list.size(); i2++) {
                ((c.a.a.z.e) list.get(i2)).f3278b.h(t, cVar);
            }
            z = true ^ list.isEmpty();
        }
        if (z) {
            invalidateSelf();
            if (t == c.a.a.o.A) {
                u(g());
            }
        }
    }

    public final void b() {
        c.a.a.d dVar = this.f3075c;
        c.a aVar = c.a.a.b0.r.f3004a;
        Rect rect = dVar.j;
        c.a.a.z.l.e eVar = new c.a.a.z.l.e(Collections.emptyList(), dVar, "__container", -1L, e.a.PRE_COMP, -1L, null, Collections.emptyList(), new c.a.a.z.j.l(null, null, null, null, null, null, null, null, null), 0, 0, 0, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, rect.width(), rect.height(), null, null, Collections.emptyList(), 1, null, false);
        c.a.a.d dVar2 = this.f3075c;
        this.p = new c.a.a.z.l.c(this, eVar, dVar2.i, dVar2);
    }

    public void c() {
        c.a.a.c0.d dVar = this.f3076d;
        if (dVar.l) {
            dVar.cancel();
        }
        this.f3075c = null;
        this.p = null;
        this.k = null;
        c.a.a.c0.d dVar2 = this.f3076d;
        dVar2.k = null;
        dVar2.i = -2.14748365E9f;
        dVar2.j = 2.14748365E9f;
        invalidateSelf();
    }

    public final void d(Canvas canvas) {
        float f2;
        float f3;
        int i2 = -1;
        if (ImageView.ScaleType.FIT_XY == this.j) {
            if (this.p == null) {
                return;
            }
            Rect bounds = getBounds();
            float width = bounds.width() / this.f3075c.j.width();
            float height = bounds.height() / this.f3075c.j.height();
            if (this.t) {
                float min = Math.min(width, height);
                if (min < 1.0f) {
                    f3 = 1.0f / min;
                    width /= f3;
                    height /= f3;
                } else {
                    f3 = 1.0f;
                }
                if (f3 > 1.0f) {
                    i2 = canvas.save();
                    float width2 = bounds.width() / 2.0f;
                    float height2 = bounds.height() / 2.0f;
                    float f4 = width2 * min;
                    float f5 = min * height2;
                    canvas.translate(width2 - f4, height2 - f5);
                    canvas.scale(f3, f3, f4, f5);
                }
            }
            this.f3074b.reset();
            this.f3074b.preScale(width, height);
            this.p.f(canvas, this.f3074b, this.q);
            if (i2 > 0) {
                canvas.restoreToCount(i2);
            }
        } else if (this.p == null) {
        } else {
            float f6 = this.f3077e;
            float min2 = Math.min(canvas.getWidth() / this.f3075c.j.width(), canvas.getHeight() / this.f3075c.j.height());
            if (f6 > min2) {
                f2 = this.f3077e / min2;
            } else {
                min2 = f6;
                f2 = 1.0f;
            }
            if (f2 > 1.0f) {
                i2 = canvas.save();
                float width3 = this.f3075c.j.width() / 2.0f;
                float height3 = this.f3075c.j.height() / 2.0f;
                float f7 = width3 * min2;
                float f8 = height3 * min2;
                float f9 = this.f3077e;
                canvas.translate((width3 * f9) - f7, (f9 * height3) - f8);
                canvas.scale(f2, f2, f7, f8);
            }
            this.f3074b.reset();
            this.f3074b.preScale(min2, min2);
            this.p.f(canvas, this.f3074b, this.q);
            if (i2 > 0) {
                canvas.restoreToCount(i2);
            }
        }
    }

    @Override // android.graphics.drawable.Drawable
    public void draw(Canvas canvas) {
        this.u = false;
        if (this.f3079g) {
            try {
                d(canvas);
            } catch (Throwable unused) {
                Objects.requireNonNull((c.a.a.c0.b) c.a.a.c0.c.f3022a);
            }
        } else {
            d(canvas);
        }
        c.a.a.c.a("Drawable#draw");
    }

    public float e() {
        return this.f3076d.e();
    }

    public float f() {
        return this.f3076d.f();
    }

    public float g() {
        return this.f3076d.d();
    }

    @Override // android.graphics.drawable.Drawable
    public int getAlpha() {
        return this.q;
    }

    @Override // android.graphics.drawable.Drawable
    public int getIntrinsicHeight() {
        c.a.a.d dVar = this.f3075c;
        if (dVar == null) {
            return -1;
        }
        return (int) (dVar.j.height() * this.f3077e);
    }

    @Override // android.graphics.drawable.Drawable
    public int getIntrinsicWidth() {
        c.a.a.d dVar = this.f3075c;
        if (dVar == null) {
            return -1;
        }
        return (int) (dVar.j.width() * this.f3077e);
    }

    @Override // android.graphics.drawable.Drawable
    public int getOpacity() {
        return -3;
    }

    public int h() {
        return this.f3076d.getRepeatCount();
    }

    public boolean i() {
        c.a.a.c0.d dVar = this.f3076d;
        if (dVar == null) {
            return false;
        }
        return dVar.l;
    }

    @Override // android.graphics.drawable.Drawable.Callback
    public void invalidateDrawable(Drawable drawable) {
        Drawable.Callback callback = getCallback();
        if (callback == null) {
            return;
        }
        callback.invalidateDrawable(this);
    }

    @Override // android.graphics.drawable.Drawable
    public void invalidateSelf() {
        if (this.u) {
            return;
        }
        this.u = true;
        Drawable.Callback callback = getCallback();
        if (callback != null) {
            callback.invalidateDrawable(this);
        }
    }

    @Override // android.graphics.drawable.Animatable
    public boolean isRunning() {
        return i();
    }

    public void j() {
        if (this.p == null) {
            this.f3080h.add(new g());
            return;
        }
        if (this.f3078f || h() == 0) {
            c.a.a.c0.d dVar = this.f3076d;
            dVar.l = true;
            boolean g2 = dVar.g();
            for (Animator.AnimatorListener animatorListener : dVar.f3020c) {
                if (Build.VERSION.SDK_INT >= 26) {
                    animatorListener.onAnimationStart(dVar, g2);
                } else {
                    animatorListener.onAnimationStart(dVar);
                }
            }
            dVar.j((int) (dVar.g() ? dVar.e() : dVar.f()));
            dVar.f3025f = 0L;
            dVar.f3027h = 0;
            dVar.h();
        }
        if (this.f3078f) {
            return;
        }
        l((int) (this.f3076d.f3023d < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? f() : e()));
        this.f3076d.c();
    }

    public void k() {
        if (this.p == null) {
            this.f3080h.add(new h());
            return;
        }
        if (this.f3078f || h() == 0) {
            c.a.a.c0.d dVar = this.f3076d;
            dVar.l = true;
            dVar.h();
            dVar.f3025f = 0L;
            if (dVar.g() && dVar.f3026g == dVar.f()) {
                dVar.f3026g = dVar.e();
            } else if (!dVar.g() && dVar.f3026g == dVar.e()) {
                dVar.f3026g = dVar.f();
            }
        }
        if (this.f3078f) {
            return;
        }
        l((int) (this.f3076d.f3023d < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? f() : e()));
        this.f3076d.c();
    }

    public void l(int i2) {
        if (this.f3075c == null) {
            this.f3080h.add(new c(i2));
        } else {
            this.f3076d.j(i2);
        }
    }

    public void m(int i2) {
        if (this.f3075c == null) {
            this.f3080h.add(new k(i2));
            return;
        }
        c.a.a.c0.d dVar = this.f3076d;
        dVar.k(dVar.i, i2 + 0.99f);
    }

    public void n(String str) {
        c.a.a.d dVar = this.f3075c;
        if (dVar == null) {
            this.f3080h.add(new n(str));
            return;
        }
        c.a.a.z.h d2 = dVar.d(str);
        if (d2 != null) {
            m((int) (d2.f3282b + d2.f3283c));
            return;
        }
        throw new IllegalArgumentException(c.b.a.a.a.r("Cannot find marker with name ", str, "."));
    }

    public void o(float f2) {
        c.a.a.d dVar = this.f3075c;
        if (dVar == null) {
            this.f3080h.add(new l(f2));
        } else {
            m((int) c.a.a.c0.f.e(dVar.k, dVar.l, f2));
        }
    }

    public void p(int i2, int i3) {
        if (this.f3075c == null) {
            this.f3080h.add(new b(i2, i3));
        } else {
            this.f3076d.k(i2, i3 + 0.99f);
        }
    }

    public void q(String str) {
        c.a.a.d dVar = this.f3075c;
        if (dVar == null) {
            this.f3080h.add(new a(str));
            return;
        }
        c.a.a.z.h d2 = dVar.d(str);
        if (d2 != null) {
            int i2 = (int) d2.f3282b;
            p(i2, ((int) d2.f3283c) + i2);
            return;
        }
        throw new IllegalArgumentException(c.b.a.a.a.r("Cannot find marker with name ", str, "."));
    }

    public void r(int i2) {
        if (this.f3075c == null) {
            this.f3080h.add(new i(i2));
            return;
        }
        c.a.a.c0.d dVar = this.f3076d;
        dVar.k(i2, (int) dVar.j);
    }

    public void s(String str) {
        c.a.a.d dVar = this.f3075c;
        if (dVar == null) {
            this.f3080h.add(new m(str));
            return;
        }
        c.a.a.z.h d2 = dVar.d(str);
        if (d2 != null) {
            r((int) d2.f3282b);
            return;
        }
        throw new IllegalArgumentException(c.b.a.a.a.r("Cannot find marker with name ", str, "."));
    }

    @Override // android.graphics.drawable.Drawable.Callback
    public void scheduleDrawable(Drawable drawable, Runnable runnable, long j) {
        Drawable.Callback callback = getCallback();
        if (callback == null) {
            return;
        }
        callback.scheduleDrawable(this, runnable, j);
    }

    @Override // android.graphics.drawable.Drawable
    public void setAlpha(int i2) {
        this.q = i2;
        invalidateSelf();
    }

    @Override // android.graphics.drawable.Drawable
    public void setColorFilter(ColorFilter colorFilter) {
        c.a.a.c0.c.b("Use addColorFilter instead.");
    }

    @Override // android.graphics.drawable.Animatable
    public void start() {
        j();
    }

    @Override // android.graphics.drawable.Animatable
    public void stop() {
        this.f3080h.clear();
        this.f3076d.c();
    }

    public void t(float f2) {
        c.a.a.d dVar = this.f3075c;
        if (dVar == null) {
            this.f3080h.add(new C0059j(f2));
        } else {
            r((int) c.a.a.c0.f.e(dVar.k, dVar.l, f2));
        }
    }

    public void u(float f2) {
        c.a.a.d dVar = this.f3075c;
        if (dVar == null) {
            this.f3080h.add(new d(f2));
            return;
        }
        this.f3076d.j(c.a.a.c0.f.e(dVar.k, dVar.l, f2));
        c.a.a.c.a("Drawable#setProgress");
    }

    @Override // android.graphics.drawable.Drawable.Callback
    public void unscheduleDrawable(Drawable drawable, Runnable runnable) {
        Drawable.Callback callback = getCallback();
        if (callback == null) {
            return;
        }
        callback.unscheduleDrawable(this, runnable);
    }

    public final void v() {
        c.a.a.d dVar = this.f3075c;
        if (dVar == null) {
            return;
        }
        float f2 = this.f3077e;
        setBounds(0, 0, (int) (dVar.j.width() * f2), (int) (this.f3075c.j.height() * f2));
    }
}