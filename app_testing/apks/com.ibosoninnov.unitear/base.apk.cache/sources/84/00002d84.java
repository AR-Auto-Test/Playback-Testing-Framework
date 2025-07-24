package g;

/* compiled from: Segment.java */
/* loaded from: classes2.dex */
public final class t {

    /* renamed from: a  reason: collision with root package name */
    public final byte[] f6209a;

    /* renamed from: b  reason: collision with root package name */
    public int f6210b;

    /* renamed from: c  reason: collision with root package name */
    public int f6211c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f6212d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f6213e;

    /* renamed from: f  reason: collision with root package name */
    public t f6214f;

    /* renamed from: g  reason: collision with root package name */
    public t f6215g;

    public t() {
        this.f6209a = new byte[8192];
        this.f6213e = true;
        this.f6212d = false;
    }

    public final t a() {
        t tVar = this.f6214f;
        t tVar2 = tVar != this ? tVar : null;
        t tVar3 = this.f6215g;
        tVar3.f6214f = tVar;
        this.f6214f.f6215g = tVar3;
        this.f6214f = null;
        this.f6215g = null;
        return tVar2;
    }

    public final t b(t tVar) {
        tVar.f6215g = this;
        tVar.f6214f = this.f6214f;
        this.f6214f.f6215g = tVar;
        this.f6214f = tVar;
        return tVar;
    }

    public final t c() {
        this.f6212d = true;
        return new t(this.f6209a, this.f6210b, this.f6211c, true, false);
    }

    public final void d(t tVar, int i) {
        if (tVar.f6213e) {
            int i2 = tVar.f6211c;
            if (i2 + i > 8192) {
                if (!tVar.f6212d) {
                    int i3 = tVar.f6210b;
                    if ((i2 + i) - i3 <= 8192) {
                        byte[] bArr = tVar.f6209a;
                        System.arraycopy(bArr, i3, bArr, 0, i2 - i3);
                        tVar.f6211c -= tVar.f6210b;
                        tVar.f6210b = 0;
                    } else {
                        throw new IllegalArgumentException();
                    }
                } else {
                    throw new IllegalArgumentException();
                }
            }
            System.arraycopy(this.f6209a, this.f6210b, tVar.f6209a, tVar.f6211c, i);
            tVar.f6211c += i;
            this.f6210b += i;
            return;
        }
        throw new IllegalArgumentException();
    }

    public t(byte[] bArr, int i, int i2, boolean z, boolean z2) {
        this.f6209a = bArr;
        this.f6210b = i;
        this.f6211c = i2;
        this.f6212d = z;
        this.f6213e = z2;
    }
}