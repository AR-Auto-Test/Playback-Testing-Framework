package c.c.a;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.util.Log;
import android.widget.ImageView;
import c.c.a.m.v.k;
import c.c.a.m.x.c.l;
import c.c.a.m.x.c.q;
import c.c.a.n.r;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: RequestBuilder.java */
/* loaded from: classes.dex */
public class h<TranscodeType> extends c.c.a.q.a<h<TranscodeType>> implements Cloneable {
    public final Context B;
    public final i C;
    public final Class<TranscodeType> D;
    public final d E;
    public j<?, ? super TranscodeType> F;
    public Object G;
    public List<c.c.a.q.e<TranscodeType>> H;
    public h<TranscodeType> I;
    public h<TranscodeType> J;
    public boolean K = true;
    public boolean L;
    public boolean M;

    /* compiled from: RequestBuilder.java */
    /* loaded from: classes.dex */
    public static /* synthetic */ class a {

        /* renamed from: a  reason: collision with root package name */
        public static final /* synthetic */ int[] f3448a;

        /* renamed from: b  reason: collision with root package name */
        public static final /* synthetic */ int[] f3449b;

        static {
            f.values();
            int[] iArr = new int[4];
            f3449b = iArr;
            try {
                iArr[3] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                f3449b[2] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                f3449b[1] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            try {
                f3449b[0] = 4;
            } catch (NoSuchFieldError unused4) {
            }
            int[] iArr2 = new int[ImageView.ScaleType.values().length];
            f3448a = iArr2;
            try {
                iArr2[ImageView.ScaleType.CENTER_CROP.ordinal()] = 1;
            } catch (NoSuchFieldError unused5) {
            }
            try {
                f3448a[ImageView.ScaleType.CENTER_INSIDE.ordinal()] = 2;
            } catch (NoSuchFieldError unused6) {
            }
            try {
                f3448a[ImageView.ScaleType.FIT_CENTER.ordinal()] = 3;
            } catch (NoSuchFieldError unused7) {
            }
            try {
                f3448a[ImageView.ScaleType.FIT_START.ordinal()] = 4;
            } catch (NoSuchFieldError unused8) {
            }
            try {
                f3448a[ImageView.ScaleType.FIT_END.ordinal()] = 5;
            } catch (NoSuchFieldError unused9) {
            }
            try {
                f3448a[ImageView.ScaleType.FIT_XY.ordinal()] = 6;
            } catch (NoSuchFieldError unused10) {
            }
            try {
                f3448a[ImageView.ScaleType.CENTER.ordinal()] = 7;
            } catch (NoSuchFieldError unused11) {
            }
            try {
                f3448a[ImageView.ScaleType.MATRIX.ordinal()] = 8;
            } catch (NoSuchFieldError unused12) {
            }
        }
    }

    static {
        new c.c.a.q.f().e(k.f3732b).k(f.LOW).o(true);
    }

    @SuppressLint({"CheckResult"})
    public h(b bVar, i iVar, Class<TranscodeType> cls, Context context) {
        c.c.a.q.f fVar;
        this.C = iVar;
        this.D = cls;
        this.B = context;
        d dVar = iVar.f3451c.f3414f;
        j<?, ?> jVar = dVar.f3431g.get(cls);
        if (jVar == null) {
            for (Map.Entry<Class<?>, j<?, ?>> entry : dVar.f3431g.entrySet()) {
                if (entry.getKey().isAssignableFrom(cls)) {
                    jVar = (j<?, ? super TranscodeType>) entry.getValue();
                }
            }
        }
        this.F = (j<?, ? super TranscodeType>) (jVar == null ? (j<?, ? super TranscodeType>) d.f3425a : jVar);
        this.E = bVar.f3414f;
        for (c.c.a.q.e<Object> eVar : iVar.k) {
            v(eVar);
        }
        synchronized (iVar) {
            fVar = iVar.l;
        }
        a(fVar);
    }

    public final <Y extends c.c.a.q.j.h<TranscodeType>> Y A(Y y, c.c.a.q.e<TranscodeType> eVar, c.c.a.q.a<?> aVar, Executor executor) {
        Objects.requireNonNull(y, "Argument must not be null");
        if (this.L) {
            c.c.a.q.c x = x(new Object(), y, eVar, null, this.F, aVar.f4129e, aVar.l, aVar.k, aVar, executor);
            c.c.a.q.c f2 = y.f();
            if (x.c(f2)) {
                if (!(!aVar.j && f2.i())) {
                    Objects.requireNonNull(f2, "Argument must not be null");
                    if (!f2.isRunning()) {
                        f2.g();
                    }
                    return y;
                }
            }
            this.C.i(y);
            y.c(x);
            i iVar = this.C;
            synchronized (iVar) {
                iVar.f3456h.f4108b.add(y);
                r rVar = iVar.f3454f;
                rVar.f4098a.add(x);
                if (!rVar.f4100c) {
                    x.g();
                } else {
                    x.clear();
                    if (Log.isLoggable("RequestTracker", 2)) {
                        Log.v("RequestTracker", "Paused, delaying request");
                    }
                    rVar.f4099b.add(x);
                }
            }
            return y;
        }
        throw new IllegalArgumentException("You must call #load() before calling #into()");
    }

    /* JADX WARN: Removed duplicated region for block: B:18:0x0085  */
    /* JADX WARN: Removed duplicated region for block: B:19:0x008b  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public c.c.a.q.j.i<ImageView, TranscodeType> B(ImageView imageView) {
        h<TranscodeType> hVar;
        Class<TranscodeType> cls;
        c.c.a.q.j.i<ImageView, TranscodeType> dVar;
        c.c.a.s.j.a();
        Objects.requireNonNull(imageView, "Argument must not be null");
        if (!c.c.a.q.a.g(this.f4126b, 2048) && this.o && imageView.getScaleType() != null) {
            switch (a.f3448a[imageView.getScaleType().ordinal()]) {
                case 1:
                    hVar = clone().h(l.f3971c, new c.c.a.m.x.c.i());
                    break;
                case 2:
                    hVar = clone().h(l.f3970b, new c.c.a.m.x.c.j());
                    hVar.z = true;
                    break;
                case 3:
                case 4:
                case 5:
                    hVar = clone().h(l.f3969a, new q());
                    hVar.z = true;
                    break;
                case 6:
                    hVar = clone().h(l.f3970b, new c.c.a.m.x.c.j());
                    hVar.z = true;
                    break;
            }
            d dVar2 = this.E;
            cls = this.D;
            Objects.requireNonNull(dVar2.f3428d);
            if (!Bitmap.class.equals(cls)) {
                dVar = new c.c.a.q.j.b(imageView);
            } else if (Drawable.class.isAssignableFrom(cls)) {
                dVar = new c.c.a.q.j.d(imageView);
            } else {
                throw new IllegalArgumentException("Unhandled class: " + cls + ", try .as*(Class).transcode(ResourceTranscoder)");
            }
            A(dVar, null, hVar, c.c.a.s.e.f4184a);
            return dVar;
        }
        hVar = this;
        d dVar22 = this.E;
        cls = this.D;
        Objects.requireNonNull(dVar22.f3428d);
        if (!Bitmap.class.equals(cls)) {
        }
        A(dVar, null, hVar, c.c.a.s.e.f4184a);
        return dVar;
    }

    public h<TranscodeType> C(c.c.a.q.e<TranscodeType> eVar) {
        if (this.w) {
            return clone().C(eVar);
        }
        this.H = null;
        return v(eVar);
    }

    public final h<TranscodeType> D(Object obj) {
        if (this.w) {
            return clone().D(obj);
        }
        this.G = obj;
        this.L = true;
        l();
        return this;
    }

    public final c.c.a.q.c E(Object obj, c.c.a.q.j.h<TranscodeType> hVar, c.c.a.q.e<TranscodeType> eVar, c.c.a.q.a<?> aVar, c.c.a.q.d dVar, j<?, ? super TranscodeType> jVar, f fVar, int i, int i2, Executor executor) {
        Context context = this.B;
        d dVar2 = this.E;
        Object obj2 = this.G;
        Class<TranscodeType> cls = this.D;
        List<c.c.a.q.e<TranscodeType>> list = this.H;
        c.c.a.m.v.l lVar = dVar2.f3432h;
        Objects.requireNonNull(jVar);
        return new c.c.a.q.h(context, dVar2, obj, obj2, cls, aVar, i, i2, fVar, hVar, eVar, list, dVar, lVar, c.c.a.q.k.a.f4166b, executor);
    }

    public h<TranscodeType> F(h<TranscodeType> hVar) {
        if (this.w) {
            return clone().F(hVar);
        }
        this.I = hVar;
        l();
        return this;
    }

    public h<TranscodeType> v(c.c.a.q.e<TranscodeType> eVar) {
        if (this.w) {
            return clone().v(eVar);
        }
        if (eVar != null) {
            if (this.H == null) {
                this.H = new ArrayList();
            }
            this.H.add(eVar);
        }
        l();
        return this;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // c.c.a.q.a
    /* renamed from: w */
    public h<TranscodeType> a(c.c.a.q.a<?> aVar) {
        Objects.requireNonNull(aVar, "Argument must not be null");
        return (h) super.a(aVar);
    }

    public final c.c.a.q.c x(Object obj, c.c.a.q.j.h<TranscodeType> hVar, c.c.a.q.e<TranscodeType> eVar, c.c.a.q.d dVar, j<?, ? super TranscodeType> jVar, f fVar, int i, int i2, c.c.a.q.a<?> aVar, Executor executor) {
        c.c.a.q.b bVar;
        c.c.a.q.b bVar2;
        c.c.a.q.i E;
        int i3;
        int i4;
        f z;
        int i5;
        int i6;
        if (this.J != null) {
            c.c.a.q.b bVar3 = new c.c.a.q.b(obj, dVar);
            bVar = bVar3;
            bVar2 = bVar3;
        } else {
            bVar = null;
            bVar2 = dVar;
        }
        h<TranscodeType> hVar2 = this.I;
        if (hVar2 != null) {
            if (!this.M) {
                j<?, ? super TranscodeType> jVar2 = hVar2.K ? jVar : hVar2.F;
                if (c.c.a.q.a.g(hVar2.f4126b, 8)) {
                    z = this.I.f4129e;
                } else {
                    z = z(fVar);
                }
                f fVar2 = z;
                h<TranscodeType> hVar3 = this.I;
                int i7 = hVar3.l;
                int i8 = hVar3.k;
                if (c.c.a.s.j.j(i, i2)) {
                    h<TranscodeType> hVar4 = this.I;
                    if (!c.c.a.s.j.j(hVar4.l, hVar4.k)) {
                        i6 = aVar.l;
                        i5 = aVar.k;
                        c.c.a.q.i iVar = new c.c.a.q.i(obj, bVar2);
                        c.c.a.q.c E2 = E(obj, hVar, eVar, aVar, iVar, jVar, fVar, i, i2, executor);
                        this.M = true;
                        h<TranscodeType> hVar5 = this.I;
                        c.c.a.q.c x = hVar5.x(obj, hVar, eVar, iVar, jVar2, fVar2, i6, i5, hVar5, executor);
                        this.M = false;
                        iVar.f4149c = E2;
                        iVar.f4150d = x;
                        E = iVar;
                    }
                }
                i5 = i8;
                i6 = i7;
                c.c.a.q.i iVar2 = new c.c.a.q.i(obj, bVar2);
                c.c.a.q.c E22 = E(obj, hVar, eVar, aVar, iVar2, jVar, fVar, i, i2, executor);
                this.M = true;
                h<TranscodeType> hVar52 = this.I;
                c.c.a.q.c x2 = hVar52.x(obj, hVar, eVar, iVar2, jVar2, fVar2, i6, i5, hVar52, executor);
                this.M = false;
                iVar2.f4149c = E22;
                iVar2.f4150d = x2;
                E = iVar2;
            } else {
                throw new IllegalStateException("You cannot use a request as both the main request and a thumbnail, consider using clone() on the request(s) passed to thumbnail()");
            }
        } else {
            E = E(obj, hVar, eVar, aVar, bVar2, jVar, fVar, i, i2, executor);
        }
        if (bVar == null) {
            return E;
        }
        h<TranscodeType> hVar6 = this.J;
        int i9 = hVar6.l;
        int i10 = hVar6.k;
        if (c.c.a.s.j.j(i, i2)) {
            h<TranscodeType> hVar7 = this.J;
            if (!c.c.a.s.j.j(hVar7.l, hVar7.k)) {
                i4 = aVar.l;
                i3 = aVar.k;
                h<TranscodeType> hVar8 = this.J;
                c.c.a.q.c x3 = hVar8.x(obj, hVar, eVar, bVar, hVar8.F, hVar8.f4129e, i4, i3, hVar8, executor);
                bVar.f4135c = E;
                bVar.f4136d = x3;
                return bVar;
            }
        }
        i3 = i10;
        i4 = i9;
        h<TranscodeType> hVar82 = this.J;
        c.c.a.q.c x32 = hVar82.x(obj, hVar, eVar, bVar, hVar82.F, hVar82.f4129e, i4, i3, hVar82, executor);
        bVar.f4135c = E;
        bVar.f4136d = x32;
        return bVar;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // c.c.a.q.a
    /* renamed from: y */
    public h<TranscodeType> clone() {
        h<TranscodeType> hVar = (h) super.clone();
        hVar.F = (j<?, ? super TranscodeType>) hVar.F.a();
        if (hVar.H != null) {
            hVar.H = new ArrayList(hVar.H);
        }
        h<TranscodeType> hVar2 = hVar.I;
        if (hVar2 != null) {
            hVar.I = hVar2.clone();
        }
        h<TranscodeType> hVar3 = hVar.J;
        if (hVar3 != null) {
            hVar.J = hVar3.clone();
        }
        return hVar;
    }

    public final f z(f fVar) {
        int ordinal = fVar.ordinal();
        if (ordinal == 0 || ordinal == 1) {
            return f.IMMEDIATE;
        }
        if (ordinal != 2) {
            if (ordinal == 3) {
                return f.NORMAL;
            }
            StringBuilder x = c.b.a.a.a.x("unknown priority: ");
            x.append(this.f4129e);
            throw new IllegalArgumentException(x.toString());
        }
        return f.HIGH;
    }
}