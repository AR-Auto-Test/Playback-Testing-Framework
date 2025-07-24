package f;

/* compiled from: RequestBody.java */
/* loaded from: classes2.dex */
public final class z extends a0 {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ t f6161a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f6162b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ byte[] f6163c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ int f6164d;

    public z(t tVar, int i, byte[] bArr, int i2) {
        this.f6161a = tVar;
        this.f6162b = i;
        this.f6163c = bArr;
        this.f6164d = i2;
    }

    @Override // f.a0
    public long a() {
        return this.f6162b;
    }

    @Override // f.a0
    public t b() {
        return this.f6161a;
    }

    @Override // f.a0
    public void c(g.f fVar) {
        fVar.write(this.f6163c, this.f6164d, this.f6162b);
    }
}