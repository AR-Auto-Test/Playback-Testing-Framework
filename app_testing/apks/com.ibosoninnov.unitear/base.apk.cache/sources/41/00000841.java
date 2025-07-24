package c.c.a.m.x.g;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.Animatable;
import android.graphics.drawable.Drawable;
import android.view.Gravity;
import c.c.a.m.t;
import c.c.a.m.x.g.g;

/* compiled from: GifDrawable.java */
/* loaded from: classes.dex */
public class c extends Drawable implements g.b, Animatable {

    /* renamed from: b  reason: collision with root package name */
    public final a f4036b;

    /* renamed from: c  reason: collision with root package name */
    public boolean f4037c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f4038d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f4039e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f4040f;

    /* renamed from: g  reason: collision with root package name */
    public int f4041g;

    /* renamed from: h  reason: collision with root package name */
    public int f4042h;
    public boolean i;
    public Paint j;
    public Rect k;

    /* compiled from: GifDrawable.java */
    /* loaded from: classes.dex */
    public static final class a extends Drawable.ConstantState {

        /* renamed from: a  reason: collision with root package name */
        public final g f4043a;

        public a(g gVar) {
            this.f4043a = gVar;
        }

        @Override // android.graphics.drawable.Drawable.ConstantState
        public int getChangingConfigurations() {
            return 0;
        }

        @Override // android.graphics.drawable.Drawable.ConstantState
        public Drawable newDrawable() {
            return new c(this);
        }

        @Override // android.graphics.drawable.Drawable.ConstantState
        public Drawable newDrawable(Resources resources) {
            return new c(this);
        }
    }

    public c(Context context, c.c.a.l.a aVar, t<Bitmap> tVar, int i, int i2, Bitmap bitmap) {
        a aVar2 = new a(new g(c.c.a.b.b(context), aVar, i, i2, tVar, bitmap));
        this.f4040f = true;
        this.f4042h = -1;
        this.f4036b = aVar2;
    }

    @Override // c.c.a.m.x.g.g.b
    public void a() {
        g gVar;
        Drawable.Callback callback = getCallback();
        while (callback instanceof Drawable) {
            callback = ((Drawable) callback).getCallback();
        }
        if (callback == null) {
            stop();
            invalidateSelf();
            return;
        }
        invalidateSelf();
        g.a aVar = this.f4036b.f4043a.i;
        if ((aVar != null ? aVar.f4054f : -1) == gVar.f4045a.c() - 1) {
            this.f4041g++;
        }
        int i = this.f4042h;
        if (i == -1 || this.f4041g < i) {
            return;
        }
        stop();
    }

    public Bitmap b() {
        return this.f4036b.f4043a.l;
    }

    public final Paint c() {
        if (this.j == null) {
            this.j = new Paint(2);
        }
        return this.j;
    }

    public final void d() {
        b.v.u.c.d(!this.f4039e, "You cannot start a recycled Drawable. Ensure thatyou clear any references to the Drawable when clearing the corresponding request.");
        if (this.f4036b.f4043a.f4045a.c() == 1) {
            invalidateSelf();
        } else if (this.f4037c) {
        } else {
            this.f4037c = true;
            g gVar = this.f4036b.f4043a;
            if (!gVar.j) {
                if (!gVar.f4047c.contains(this)) {
                    boolean isEmpty = gVar.f4047c.isEmpty();
                    gVar.f4047c.add(this);
                    if (isEmpty && !gVar.f4050f) {
                        gVar.f4050f = true;
                        gVar.j = false;
                        gVar.a();
                    }
                    invalidateSelf();
                    return;
                }
                throw new IllegalStateException("Cannot subscribe twice in a row");
            }
            throw new IllegalStateException("Cannot subscribe to a cleared frame loader");
        }
    }

    @Override // android.graphics.drawable.Drawable
    public void draw(Canvas canvas) {
        Bitmap bitmap;
        if (this.f4039e) {
            return;
        }
        if (this.i) {
            int intrinsicWidth = getIntrinsicWidth();
            int intrinsicHeight = getIntrinsicHeight();
            Rect bounds = getBounds();
            if (this.k == null) {
                this.k = new Rect();
            }
            Gravity.apply(119, intrinsicWidth, intrinsicHeight, bounds, this.k);
            this.i = false;
        }
        g gVar = this.f4036b.f4043a;
        g.a aVar = gVar.i;
        if (aVar != null) {
            bitmap = aVar.f4056h;
        } else {
            bitmap = gVar.l;
        }
        if (this.k == null) {
            this.k = new Rect();
        }
        canvas.drawBitmap(bitmap, (Rect) null, this.k, c());
    }

    public final void e() {
        this.f4037c = false;
        g gVar = this.f4036b.f4043a;
        gVar.f4047c.remove(this);
        if (gVar.f4047c.isEmpty()) {
            gVar.f4050f = false;
        }
    }

    @Override // android.graphics.drawable.Drawable
    public Drawable.ConstantState getConstantState() {
        return this.f4036b;
    }

    @Override // android.graphics.drawable.Drawable
    public int getIntrinsicHeight() {
        return this.f4036b.f4043a.q;
    }

    @Override // android.graphics.drawable.Drawable
    public int getIntrinsicWidth() {
        return this.f4036b.f4043a.p;
    }

    @Override // android.graphics.drawable.Drawable
    public int getOpacity() {
        return -2;
    }

    @Override // android.graphics.drawable.Animatable
    public boolean isRunning() {
        return this.f4037c;
    }

    @Override // android.graphics.drawable.Drawable
    public void onBoundsChange(Rect rect) {
        super.onBoundsChange(rect);
        this.i = true;
    }

    @Override // android.graphics.drawable.Drawable
    public void setAlpha(int i) {
        c().setAlpha(i);
    }

    @Override // android.graphics.drawable.Drawable
    public void setColorFilter(ColorFilter colorFilter) {
        c().setColorFilter(colorFilter);
    }

    @Override // android.graphics.drawable.Drawable
    public boolean setVisible(boolean z, boolean z2) {
        b.v.u.c.d(!this.f4039e, "Cannot change the visibility of a recycled resource. Ensure that you unset the Drawable from your View before changing the View's visibility.");
        this.f4040f = z;
        if (!z) {
            e();
        } else if (this.f4038d) {
            d();
        }
        return super.setVisible(z, z2);
    }

    @Override // android.graphics.drawable.Animatable
    public void start() {
        this.f4038d = true;
        this.f4041g = 0;
        if (this.f4040f) {
            d();
        }
    }

    @Override // android.graphics.drawable.Animatable
    public void stop() {
        this.f4038d = false;
        e();
    }

    public c(a aVar) {
        this.f4040f = true;
        this.f4042h = -1;
        this.f4036b = aVar;
    }
}