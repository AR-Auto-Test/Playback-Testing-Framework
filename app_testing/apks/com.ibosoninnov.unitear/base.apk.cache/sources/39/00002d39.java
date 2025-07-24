package f.g0.i;

/* compiled from: Settings.java */
/* loaded from: classes2.dex */
public final class t {

    /* renamed from: a  reason: collision with root package name */
    public int f6004a;

    /* renamed from: b  reason: collision with root package name */
    public final int[] f6005b = new int[10];

    public int a() {
        if ((this.f6004a & 128) != 0) {
            return this.f6005b[7];
        }
        return 65535;
    }

    public t b(int i, int i2) {
        if (i >= 0) {
            int[] iArr = this.f6005b;
            if (i < iArr.length) {
                this.f6004a = (1 << i) | this.f6004a;
                iArr[i] = i2;
            }
        }
        return this;
    }
}