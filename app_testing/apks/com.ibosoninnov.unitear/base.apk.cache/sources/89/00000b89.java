package com.airbnb.lottie;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Bitmap;
import android.graphics.PathMeasure;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Parcel;
import android.os.Parcelable;
import android.provider.Settings;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.view.View;
import android.widget.ImageView;
import b.b.h.n;
import b.j.j.q;
import c.a.a.c0.g;
import c.a.a.f;
import c.a.a.h;
import c.a.a.i;
import c.a.a.j;
import c.a.a.l;
import c.a.a.o;
import c.a.a.r;
import c.a.a.s;
import c.a.a.t;
import c.a.a.u;
import c.a.a.v;
import c.a.a.w;
import c.a.a.z.e;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.io.ByteArrayInputStream;
import java.io.InterruptedIOException;
import java.lang.ref.WeakReference;
import java.net.ProtocolException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.net.UnknownServiceException;
import java.nio.channels.ClosedChannelException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import javax.net.ssl.SSLException;

/* loaded from: classes.dex */
public class LottieAnimationView extends n {

    /* renamed from: b  reason: collision with root package name */
    public static final String f5518b = LottieAnimationView.class.getSimpleName();

    /* renamed from: c  reason: collision with root package name */
    public static final l<Throwable> f5519c = new a();

    /* renamed from: d  reason: collision with root package name */
    public final l<c.a.a.d> f5520d;

    /* renamed from: e  reason: collision with root package name */
    public final l<Throwable> f5521e;

    /* renamed from: f  reason: collision with root package name */
    public l<Throwable> f5522f;

    /* renamed from: g  reason: collision with root package name */
    public int f5523g;

    /* renamed from: h  reason: collision with root package name */
    public final j f5524h;
    public boolean i;
    public String j;
    public int k;
    public boolean l;
    public boolean m;
    public boolean n;
    public boolean o;
    public boolean p;
    public u q;
    public Set<c.a.a.n> r;
    public int s;
    public r<c.a.a.d> t;
    public c.a.a.d u;

    /* loaded from: classes.dex */
    public class a implements l<Throwable> {
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // c.a.a.l
        public void a(Throwable th) {
            Throwable th2 = th;
            PathMeasure pathMeasure = g.f3031a;
            if ((th2 instanceof SocketException) || (th2 instanceof ClosedChannelException) || (th2 instanceof InterruptedIOException) || (th2 instanceof ProtocolException) || (th2 instanceof SSLException) || (th2 instanceof UnknownHostException) || (th2 instanceof UnknownServiceException)) {
                c.a.a.c0.c.c("Unable to load composition.", th2);
                return;
            }
            throw new IllegalStateException("Unable to parse composition", th2);
        }
    }

    /* loaded from: classes.dex */
    public class b implements l<c.a.a.d> {
        public b() {
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // c.a.a.l
        public void a(c.a.a.d dVar) {
            LottieAnimationView.this.setComposition(dVar);
        }
    }

    /* loaded from: classes.dex */
    public class c implements l<Throwable> {
        public c() {
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // c.a.a.l
        public void a(Throwable th) {
            Throwable th2 = th;
            LottieAnimationView lottieAnimationView = LottieAnimationView.this;
            int i = lottieAnimationView.f5523g;
            if (i != 0) {
                lottieAnimationView.setImageResource(i);
            }
            l<Throwable> lVar = LottieAnimationView.this.f5522f;
            if (lVar == null) {
                String str = LottieAnimationView.f5518b;
                lVar = LottieAnimationView.f5519c;
            }
            lVar.a(th2);
        }
    }

    /* loaded from: classes.dex */
    public static class d extends View.BaseSavedState {
        public static final Parcelable.Creator<d> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public String f5527b;

        /* renamed from: c  reason: collision with root package name */
        public int f5528c;

        /* renamed from: d  reason: collision with root package name */
        public float f5529d;

        /* renamed from: e  reason: collision with root package name */
        public boolean f5530e;

        /* renamed from: f  reason: collision with root package name */
        public String f5531f;

        /* renamed from: g  reason: collision with root package name */
        public int f5532g;

        /* renamed from: h  reason: collision with root package name */
        public int f5533h;

        /* loaded from: classes.dex */
        public class a implements Parcelable.Creator<d> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.Creator
            public d createFromParcel(Parcel parcel) {
                return new d(parcel, null);
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
            @Override // android.os.Parcelable.Creator
            public d[] newArray(int i) {
                return new d[i];
            }
        }

        public d(Parcelable parcelable) {
            super(parcelable);
        }

        @Override // android.view.View.BaseSavedState, android.view.AbsSavedState, android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            super.writeToParcel(parcel, i);
            parcel.writeString(this.f5527b);
            parcel.writeFloat(this.f5529d);
            parcel.writeInt(this.f5530e ? 1 : 0);
            parcel.writeString(this.f5531f);
            parcel.writeInt(this.f5532g);
            parcel.writeInt(this.f5533h);
        }

        public d(Parcel parcel, a aVar) {
            super(parcel);
            this.f5527b = parcel.readString();
            this.f5529d = parcel.readFloat();
            this.f5530e = parcel.readInt() == 1;
            this.f5531f = parcel.readString();
            this.f5532g = parcel.readInt();
            this.f5533h = parcel.readInt();
        }
    }

    public LottieAnimationView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        String string;
        this.f5520d = new b();
        this.f5521e = new c();
        this.f5523g = 0;
        j jVar = new j();
        this.f5524h = jVar;
        this.l = false;
        this.m = false;
        this.n = false;
        this.o = false;
        this.p = true;
        this.q = u.AUTOMATIC;
        this.r = new HashSet();
        this.s = 0;
        TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, t.f3134a);
        if (!isInEditMode()) {
            this.p = obtainStyledAttributes.getBoolean(1, true);
            boolean hasValue = obtainStyledAttributes.hasValue(9);
            boolean hasValue2 = obtainStyledAttributes.hasValue(5);
            boolean hasValue3 = obtainStyledAttributes.hasValue(15);
            if (hasValue && hasValue2) {
                throw new IllegalArgumentException("lottie_rawRes and lottie_fileName cannot be used at the same time. Please use only one at once.");
            }
            if (hasValue) {
                int resourceId = obtainStyledAttributes.getResourceId(9, 0);
                if (resourceId != 0) {
                    setAnimation(resourceId);
                }
            } else if (hasValue2) {
                String string2 = obtainStyledAttributes.getString(5);
                if (string2 != null) {
                    setAnimation(string2);
                }
            } else if (hasValue3 && (string = obtainStyledAttributes.getString(15)) != null) {
                setAnimationFromUrl(string);
            }
            setFallbackResource(obtainStyledAttributes.getResourceId(4, 0));
        }
        if (obtainStyledAttributes.getBoolean(0, false)) {
            this.n = true;
            this.o = true;
        }
        if (obtainStyledAttributes.getBoolean(7, false)) {
            jVar.f3076d.setRepeatCount(-1);
        }
        if (obtainStyledAttributes.hasValue(12)) {
            setRepeatMode(obtainStyledAttributes.getInt(12, 1));
        }
        if (obtainStyledAttributes.hasValue(11)) {
            setRepeatCount(obtainStyledAttributes.getInt(11, -1));
        }
        if (obtainStyledAttributes.hasValue(14)) {
            setSpeed(obtainStyledAttributes.getFloat(14, 1.0f));
        }
        setImageAssetsFolder(obtainStyledAttributes.getString(6));
        setProgress(obtainStyledAttributes.getFloat(8, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD));
        boolean z = obtainStyledAttributes.getBoolean(3, false);
        if (jVar.o != z) {
            jVar.o = z;
            if (jVar.f3075c != null) {
                jVar.b();
            }
        }
        if (obtainStyledAttributes.hasValue(2)) {
            jVar.a(new e("**"), o.C, new c.a.a.d0.c(new v(obtainStyledAttributes.getColor(2, 0))));
        }
        if (obtainStyledAttributes.hasValue(13)) {
            jVar.f3077e = obtainStyledAttributes.getFloat(13, 1.0f);
            jVar.v();
        }
        if (obtainStyledAttributes.hasValue(10)) {
            int i = obtainStyledAttributes.getInt(10, 0);
            u.values();
            setRenderMode(u.values()[i >= 3 ? 0 : i]);
        }
        if (getScaleType() != null) {
            jVar.j = getScaleType();
        }
        obtainStyledAttributes.recycle();
        Context context2 = getContext();
        PathMeasure pathMeasure = g.f3031a;
        Boolean valueOf = Boolean.valueOf(Settings.Global.getFloat(context2.getContentResolver(), "animator_duration_scale", 1.0f) != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        Objects.requireNonNull(jVar);
        jVar.f3078f = valueOf.booleanValue();
        d();
        this.i = true;
    }

    private void setCompositionTask(r<c.a.a.d> rVar) {
        this.u = null;
        this.f5524h.c();
        c();
        rVar.b(this.f5520d);
        rVar.a(this.f5521e);
        this.t = rVar;
    }

    @Override // android.view.View
    public void buildDrawingCache(boolean z) {
        this.s++;
        super.buildDrawingCache(z);
        if (this.s == 1 && getWidth() > 0 && getHeight() > 0 && getLayerType() == 1 && getDrawingCache(z) == null) {
            setRenderMode(u.HARDWARE);
        }
        this.s--;
        c.a.a.c.a("buildDrawingCache");
    }

    public final void c() {
        r<c.a.a.d> rVar = this.t;
        if (rVar != null) {
            l<c.a.a.d> lVar = this.f5520d;
            synchronized (rVar) {
                rVar.f3126b.remove(lVar);
            }
            r<c.a.a.d> rVar2 = this.t;
            l<Throwable> lVar2 = this.f5521e;
            synchronized (rVar2) {
                rVar2.f3127c.remove(lVar2);
            }
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:18:0x0027, code lost:
        if (r3 != false) goto L5;
     */
    /* JADX WARN: Code restructure failed: missing block: B:4:0x000a, code lost:
        if (r0 != 1) goto L4;
     */
    /* JADX WARN: Code restructure failed: missing block: B:5:0x000c, code lost:
        r1 = 1;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void d() {
        int ordinal = this.q.ordinal();
        int i = 2;
        if (ordinal == 0) {
            c.a.a.d dVar = this.u;
            boolean z = false;
            if ((dVar == null || !dVar.n || Build.VERSION.SDK_INT >= 28) && (dVar == null || dVar.o <= 4)) {
                z = true;
            }
        }
        if (i != getLayerType()) {
            setLayerType(i, null);
        }
    }

    public void e() {
        if (isShown()) {
            this.f5524h.j();
            d();
            return;
        }
        this.l = true;
    }

    public c.a.a.d getComposition() {
        return this.u;
    }

    public long getDuration() {
        c.a.a.d dVar = this.u;
        if (dVar != null) {
            return dVar.b();
        }
        return 0L;
    }

    public int getFrame() {
        return (int) this.f5524h.f3076d.f3026g;
    }

    public String getImageAssetsFolder() {
        return this.f5524h.l;
    }

    public float getMaxFrame() {
        return this.f5524h.e();
    }

    public float getMinFrame() {
        return this.f5524h.f();
    }

    public s getPerformanceTracker() {
        c.a.a.d dVar = this.f5524h.f3075c;
        if (dVar != null) {
            return dVar.f3037a;
        }
        return null;
    }

    public float getProgress() {
        return this.f5524h.g();
    }

    public int getRepeatCount() {
        return this.f5524h.h();
    }

    public int getRepeatMode() {
        return this.f5524h.f3076d.getRepeatMode();
    }

    public float getScale() {
        return this.f5524h.f3077e;
    }

    public float getSpeed() {
        return this.f5524h.f3076d.f3023d;
    }

    @Override // android.widget.ImageView, android.view.View, android.graphics.drawable.Drawable.Callback
    public void invalidateDrawable(Drawable drawable) {
        Drawable drawable2 = getDrawable();
        j jVar = this.f5524h;
        if (drawable2 == jVar) {
            super.invalidateDrawable(jVar);
        } else {
            super.invalidateDrawable(drawable);
        }
    }

    @Override // android.widget.ImageView, android.view.View
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        if (this.o || this.n) {
            e();
            this.o = false;
            this.n = false;
        }
    }

    @Override // android.widget.ImageView, android.view.View
    public void onDetachedFromWindow() {
        if (this.f5524h.i()) {
            this.n = false;
            this.m = false;
            this.l = false;
            j jVar = this.f5524h;
            jVar.f3080h.clear();
            jVar.f3076d.cancel();
            d();
            this.n = true;
        }
        super.onDetachedFromWindow();
    }

    @Override // android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        if (!(parcelable instanceof d)) {
            super.onRestoreInstanceState(parcelable);
            return;
        }
        d dVar = (d) parcelable;
        super.onRestoreInstanceState(dVar.getSuperState());
        String str = dVar.f5527b;
        this.j = str;
        if (!TextUtils.isEmpty(str)) {
            setAnimation(this.j);
        }
        int i = dVar.f5528c;
        this.k = i;
        if (i != 0) {
            setAnimation(i);
        }
        setProgress(dVar.f5529d);
        if (dVar.f5530e) {
            e();
        }
        this.f5524h.l = dVar.f5531f;
        setRepeatMode(dVar.f5532g);
        setRepeatCount(dVar.f5533h);
    }

    @Override // android.view.View
    public Parcelable onSaveInstanceState() {
        boolean z;
        d dVar = new d(super.onSaveInstanceState());
        dVar.f5527b = this.j;
        dVar.f5528c = this.k;
        dVar.f5529d = this.f5524h.g();
        if (!this.f5524h.i()) {
            AtomicInteger atomicInteger = q.f2214a;
            if (isAttachedToWindow() || !this.n) {
                z = false;
                dVar.f5530e = z;
                j jVar = this.f5524h;
                dVar.f5531f = jVar.l;
                dVar.f5532g = jVar.f3076d.getRepeatMode();
                dVar.f5533h = this.f5524h.h();
                return dVar;
            }
        }
        z = true;
        dVar.f5530e = z;
        j jVar2 = this.f5524h;
        dVar.f5531f = jVar2.l;
        dVar.f5532g = jVar2.f3076d.getRepeatMode();
        dVar.f5533h = this.f5524h.h();
        return dVar;
    }

    @Override // android.view.View
    public void onVisibilityChanged(View view, int i) {
        if (this.i) {
            if (isShown()) {
                if (this.m) {
                    if (isShown()) {
                        this.f5524h.k();
                        d();
                    } else {
                        this.l = false;
                        this.m = true;
                    }
                } else if (this.l) {
                    e();
                }
                this.m = false;
                this.l = false;
            } else if (this.f5524h.i()) {
                this.o = false;
                this.n = false;
                this.m = false;
                this.l = false;
                j jVar = this.f5524h;
                jVar.f3080h.clear();
                jVar.f3076d.i();
                d();
                this.m = true;
            }
        }
    }

    public void setAnimation(int i) {
        r<c.a.a.d> a2;
        this.k = i;
        this.j = null;
        if (this.p) {
            Context context = getContext();
            a2 = c.a.a.e.a(c.a.a.e.f(context, i), new h(new WeakReference(context), context.getApplicationContext(), i));
        } else {
            Context context2 = getContext();
            Map<String, r<c.a.a.d>> map = c.a.a.e.f3059a;
            a2 = c.a.a.e.a(null, new h(new WeakReference(context2), context2.getApplicationContext(), i));
        }
        setCompositionTask(a2);
    }

    @Deprecated
    public void setAnimationFromJson(String str) {
        setCompositionTask(c.a.a.e.a(null, new i(new ByteArrayInputStream(str.getBytes()), null)));
    }

    public void setAnimationFromUrl(String str) {
        r<c.a.a.d> a2;
        if (this.p) {
            Context context = getContext();
            Map<String, r<c.a.a.d>> map = c.a.a.e.f3059a;
            String q = c.b.a.a.a.q("url_", str);
            a2 = c.a.a.e.a(q, new f(context, str, q));
        } else {
            a2 = c.a.a.e.a(null, new f(getContext(), str, null));
        }
        setCompositionTask(a2);
    }

    public void setApplyingOpacityToLayersEnabled(boolean z) {
        this.f5524h.s = z;
    }

    public void setCacheComposition(boolean z) {
        this.p = z;
    }

    public void setComposition(c.a.a.d dVar) {
        this.f5524h.setCallback(this);
        this.u = dVar;
        j jVar = this.f5524h;
        if (jVar.f3075c != dVar) {
            jVar.u = false;
            jVar.c();
            jVar.f3075c = dVar;
            jVar.b();
            c.a.a.c0.d dVar2 = jVar.f3076d;
            r2 = dVar2.k == null;
            dVar2.k = dVar;
            if (r2) {
                dVar2.k((int) Math.max(dVar2.i, dVar.k), (int) Math.min(dVar2.j, dVar.l));
            } else {
                dVar2.k((int) dVar.k, (int) dVar.l);
            }
            float f2 = dVar2.f3026g;
            dVar2.f3026g = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            dVar2.j((int) f2);
            dVar2.b();
            jVar.u(jVar.f3076d.getAnimatedFraction());
            jVar.f3077e = jVar.f3077e;
            jVar.v();
            jVar.v();
            Iterator it = new ArrayList(jVar.f3080h).iterator();
            while (it.hasNext()) {
                ((j.o) it.next()).a(dVar);
                it.remove();
            }
            jVar.f3080h.clear();
            dVar.f3037a.f3131a = jVar.r;
            Drawable.Callback callback = jVar.getCallback();
            if (callback instanceof ImageView) {
                ImageView imageView = (ImageView) callback;
                imageView.setImageDrawable(null);
                imageView.setImageDrawable(jVar);
            }
            r2 = true;
        }
        d();
        if (getDrawable() != this.f5524h || r2) {
            onVisibilityChanged(this, getVisibility());
            requestLayout();
            for (c.a.a.n nVar : this.r) {
                nVar.a(dVar);
            }
        }
    }

    public void setFailureListener(l<Throwable> lVar) {
        this.f5522f = lVar;
    }

    public void setFallbackResource(int i) {
        this.f5523g = i;
    }

    public void setFontAssetDelegate(c.a.a.a aVar) {
        c.a.a.y.a aVar2 = this.f5524h.n;
    }

    public void setFrame(int i) {
        this.f5524h.l(i);
    }

    public void setImageAssetDelegate(c.a.a.b bVar) {
        j jVar = this.f5524h;
        jVar.m = bVar;
        c.a.a.y.b bVar2 = jVar.k;
        if (bVar2 != null) {
            bVar2.f3256d = bVar;
        }
    }

    public void setImageAssetsFolder(String str) {
        this.f5524h.l = str;
    }

    @Override // b.b.h.n, android.widget.ImageView
    public void setImageBitmap(Bitmap bitmap) {
        c();
        super.setImageBitmap(bitmap);
    }

    @Override // b.b.h.n, android.widget.ImageView
    public void setImageDrawable(Drawable drawable) {
        c();
        super.setImageDrawable(drawable);
    }

    @Override // b.b.h.n, android.widget.ImageView
    public void setImageResource(int i) {
        c();
        super.setImageResource(i);
    }

    public void setMaxFrame(int i) {
        this.f5524h.m(i);
    }

    public void setMaxProgress(float f2) {
        this.f5524h.o(f2);
    }

    public void setMinAndMaxFrame(String str) {
        this.f5524h.q(str);
    }

    public void setMinFrame(int i) {
        this.f5524h.r(i);
    }

    public void setMinProgress(float f2) {
        this.f5524h.t(f2);
    }

    public void setPerformanceTrackingEnabled(boolean z) {
        j jVar = this.f5524h;
        jVar.r = z;
        c.a.a.d dVar = jVar.f3075c;
        if (dVar != null) {
            dVar.f3037a.f3131a = z;
        }
    }

    public void setProgress(float f2) {
        this.f5524h.u(f2);
    }

    public void setRenderMode(u uVar) {
        this.q = uVar;
        d();
    }

    public void setRepeatCount(int i) {
        this.f5524h.f3076d.setRepeatCount(i);
    }

    public void setRepeatMode(int i) {
        this.f5524h.f3076d.setRepeatMode(i);
    }

    public void setSafeMode(boolean z) {
        this.f5524h.f3079g = z;
    }

    public void setScale(float f2) {
        j jVar = this.f5524h;
        jVar.f3077e = f2;
        jVar.v();
        if (getDrawable() == this.f5524h) {
            setImageDrawable(null);
            setImageDrawable(this.f5524h);
        }
    }

    @Override // android.widget.ImageView
    public void setScaleType(ImageView.ScaleType scaleType) {
        super.setScaleType(scaleType);
        j jVar = this.f5524h;
        if (jVar != null) {
            jVar.j = scaleType;
        }
    }

    public void setSpeed(float f2) {
        this.f5524h.f3076d.f3023d = f2;
    }

    public void setTextDelegate(w wVar) {
        Objects.requireNonNull(this.f5524h);
    }

    public void setMaxFrame(String str) {
        this.f5524h.n(str);
    }

    public void setMinFrame(String str) {
        this.f5524h.s(str);
    }

    public void setAnimation(String str) {
        r<c.a.a.d> a2;
        this.j = str;
        this.k = 0;
        if (this.p) {
            Context context = getContext();
            Map<String, r<c.a.a.d>> map = c.a.a.e.f3059a;
            String q = c.b.a.a.a.q("asset_", str);
            a2 = c.a.a.e.a(q, new c.a.a.g(context.getApplicationContext(), str, q));
        } else {
            Context context2 = getContext();
            Map<String, r<c.a.a.d>> map2 = c.a.a.e.f3059a;
            a2 = c.a.a.e.a(null, new c.a.a.g(context2.getApplicationContext(), str, null));
        }
        setCompositionTask(a2);
    }
}