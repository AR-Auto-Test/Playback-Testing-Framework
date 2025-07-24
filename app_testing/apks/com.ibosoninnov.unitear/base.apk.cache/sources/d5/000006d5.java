package c.a.a.z.l;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.drawable.Drawable;
import android.text.TextUtils;
import android.util.Base64;
import android.view.View;
import c.a.a.j;
import c.a.a.k;
import c.a.a.o;
import c.a.a.x.c.p;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.io.IOException;

/* compiled from: ImageLayer.java */
/* loaded from: classes.dex */
public class d extends b {
    public c.a.a.x.c.a<ColorFilter, ColorFilter> A;
    public final Paint x;
    public final Rect y;
    public final Rect z;

    public d(j jVar, e eVar) {
        super(jVar, eVar);
        this.x = new c.a.a.x.a(3);
        this.y = new Rect();
        this.z = new Rect();
    }

    @Override // c.a.a.z.l.b, c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        super.d(rectF, matrix, z);
        Bitmap r = r();
        if (r != null) {
            rectF.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, c.a.a.c0.g.c() * r.getWidth(), c.a.a.c0.g.c() * r.getHeight());
            this.m.mapRect(rectF);
        }
    }

    @Override // c.a.a.z.l.b, c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        this.v.c(t, cVar);
        if (t == o.C) {
            if (cVar == null) {
                this.A = null;
            } else {
                this.A = new p(cVar, null);
            }
        }
    }

    @Override // c.a.a.z.l.b
    public void k(Canvas canvas, Matrix matrix, int i) {
        Bitmap r = r();
        if (r == null || r.isRecycled()) {
            return;
        }
        float c2 = c.a.a.c0.g.c();
        this.x.setAlpha(i);
        c.a.a.x.c.a<ColorFilter, ColorFilter> aVar = this.A;
        if (aVar != null) {
            this.x.setColorFilter(aVar.e());
        }
        canvas.save();
        canvas.concat(matrix);
        this.y.set(0, 0, r.getWidth(), r.getHeight());
        this.z.set(0, 0, (int) (r.getWidth() * c2), (int) (r.getHeight() * c2));
        canvas.drawBitmap(r, this.y, this.z, this.x);
        canvas.restore();
    }

    public final Bitmap r() {
        c.a.a.y.b bVar;
        k kVar;
        String str = this.o.f3401g;
        j jVar = this.n;
        if (jVar.getCallback() == null) {
            bVar = null;
        } else {
            c.a.a.y.b bVar2 = jVar.k;
            if (bVar2 != null) {
                Drawable.Callback callback = jVar.getCallback();
                Context context = (callback != null && (callback instanceof View)) ? ((View) callback).getContext() : null;
                if (!((context == null && bVar2.f3254b == null) || bVar2.f3254b.equals(context))) {
                    jVar.k = null;
                }
            }
            if (jVar.k == null) {
                jVar.k = new c.a.a.y.b(jVar.getCallback(), jVar.l, jVar.m, jVar.f3075c.f3040d);
            }
            bVar = jVar.k;
        }
        if (bVar == null || (kVar = bVar.f3257e.get(str)) == null) {
            return null;
        }
        Bitmap bitmap = kVar.f3113e;
        if (bitmap != null) {
            return bitmap;
        }
        c.a.a.b bVar3 = bVar.f3256d;
        if (bVar3 != null) {
            Bitmap a2 = bVar3.a(kVar);
            if (a2 != null) {
                bVar.a(str, a2);
                return a2;
            }
            return a2;
        }
        String str2 = kVar.f3112d;
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = true;
        options.inDensity = 160;
        if (str2.startsWith("data:") && str2.indexOf("base64,") > 0) {
            try {
                byte[] decode = Base64.decode(str2.substring(str2.indexOf(44) + 1), 0);
                Bitmap decodeByteArray = BitmapFactory.decodeByteArray(decode, 0, decode.length, options);
                bVar.a(str, decodeByteArray);
                return decodeByteArray;
            } catch (IllegalArgumentException e2) {
                c.a.a.c0.c.c("data URL did not have correct base64 format.", e2);
                return null;
            }
        }
        try {
            if (!TextUtils.isEmpty(bVar.f3255c)) {
                Bitmap e3 = c.a.a.c0.g.e(BitmapFactory.decodeStream(bVar.f3254b.getAssets().open(bVar.f3255c + str2), null, options), kVar.f3109a, kVar.f3110b);
                bVar.a(str, e3);
                return e3;
            }
            throw new IllegalStateException("You must set an images folder before loading an image. Set it with LottieComposition#setImagesFolder or LottieDrawable#setImagesFolder");
        } catch (IOException e4) {
            c.a.a.c0.c.c("Unable to open asset.", e4);
            return null;
        }
    }
}