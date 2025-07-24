package f;

import f.r;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;

/* compiled from: RealCall.java */
/* loaded from: classes2.dex */
public final class x implements d {

    /* renamed from: b  reason: collision with root package name */
    public final v f6142b;

    /* renamed from: c  reason: collision with root package name */
    public final f.g0.g.h f6143c;

    /* renamed from: d  reason: collision with root package name */
    public n f6144d;

    /* renamed from: e  reason: collision with root package name */
    public final y f6145e;

    /* renamed from: f  reason: collision with root package name */
    public final boolean f6146f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f6147g;

    /* compiled from: RealCall.java */
    /* loaded from: classes2.dex */
    public final class a extends f.g0.b {

        /* renamed from: c  reason: collision with root package name */
        public final e f6148c;

        public a(e eVar) {
            super("OkHttp %s", x.this.d());
            this.f6148c = eVar;
        }

        @Override // f.g0.b
        public void a() {
            boolean z;
            b0 c2;
            try {
                try {
                    c2 = x.this.c();
                } catch (IOException e2) {
                    e = e2;
                    z = false;
                }
                try {
                    if (x.this.f6143c.f5840e) {
                        this.f6148c.b(x.this, new IOException("Canceled"));
                    } else {
                        this.f6148c.a(x.this, c2);
                    }
                } catch (IOException e3) {
                    e = e3;
                    z = true;
                    if (z) {
                        f.g0.j.f fVar = f.g0.j.f.f6032a;
                        fVar.k(4, "Callback failure for " + x.this.e(), e);
                    } else {
                        Objects.requireNonNull(x.this.f6144d);
                        this.f6148c.b(x.this, e);
                    }
                    x.this.f6142b.f6122d.c(this);
                }
                x.this.f6142b.f6122d.c(this);
            } catch (Throwable th) {
                x.this.f6142b.f6122d.c(this);
                throw th;
            }
        }
    }

    public x(v vVar, y yVar, boolean z) {
        this.f6142b = vVar;
        this.f6145e = yVar;
        this.f6146f = z;
        this.f6143c = new f.g0.g.h(vVar, z);
    }

    public void a() {
        f.g0.g.c cVar;
        f.g0.f.c cVar2;
        f.g0.g.h hVar = this.f6143c;
        hVar.f5840e = true;
        f.g0.f.g gVar = hVar.f5838c;
        if (gVar != null) {
            synchronized (gVar.f5813d) {
                gVar.m = true;
                cVar = gVar.n;
                cVar2 = gVar.j;
            }
            if (cVar != null) {
                cVar.cancel();
            } else if (cVar2 != null) {
                f.g0.c.g(cVar2.f5792d);
            }
        }
    }

    public void b(e eVar) {
        synchronized (this) {
            if (!this.f6147g) {
                this.f6147g = true;
            } else {
                throw new IllegalStateException("Already Executed");
            }
        }
        this.f6143c.f5839d = f.g0.j.f.f6032a.i("response.body().close()");
        Objects.requireNonNull(this.f6144d);
        l lVar = this.f6142b.f6122d;
        a aVar = new a(eVar);
        synchronized (lVar) {
            if (lVar.f6075c.size() < 64 && lVar.e(aVar) < 5) {
                lVar.f6075c.add(aVar);
                lVar.b().execute(aVar);
            } else {
                lVar.f6074b.add(aVar);
            }
        }
    }

    public b0 c() {
        ArrayList arrayList = new ArrayList();
        arrayList.addAll(this.f6142b.f6125g);
        arrayList.add(this.f6143c);
        arrayList.add(new f.g0.g.a(this.f6142b.k));
        Objects.requireNonNull(this.f6142b);
        arrayList.add(new f.g0.e.a(null));
        arrayList.add(new f.g0.f.a(this.f6142b));
        if (!this.f6146f) {
            arrayList.addAll(this.f6142b.f6126h);
        }
        arrayList.add(new f.g0.g.b(this.f6146f));
        y yVar = this.f6145e;
        n nVar = this.f6144d;
        v vVar = this.f6142b;
        return new f.g0.g.f(arrayList, null, null, null, 0, yVar, this, nVar, vVar.x, vVar.y, vVar.z).a(yVar);
    }

    public Object clone() {
        v vVar = this.f6142b;
        x xVar = new x(vVar, this.f6145e, this.f6146f);
        xVar.f6144d = ((o) vVar.i).f6079a;
        return xVar;
    }

    public String d() {
        r rVar = this.f6145e.f6150a;
        Objects.requireNonNull(rVar);
        r.a aVar = new r.a();
        if (aVar.c(rVar, "/...") != 1) {
            aVar = null;
        }
        Objects.requireNonNull(aVar);
        aVar.f6095b = r.b("", " \"':;<=>@[]^`{}|/\\?#", false, false, false, true);
        aVar.f6096c = r.b("", " \"':;<=>@[]^`{}|/\\?#", false, false, false, true);
        return aVar.a().j;
    }

    public String e() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.f6143c.f5840e ? "canceled " : "");
        sb.append(this.f6146f ? "web socket" : "call");
        sb.append(" to ");
        sb.append(d());
        return sb.toString();
    }
}