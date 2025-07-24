package b.b.h;

/* compiled from: RtlSpacingHelper.java */
/* loaded from: classes.dex */
public class p0 {

    /* renamed from: a  reason: collision with root package name */
    public int f897a = 0;

    /* renamed from: b  reason: collision with root package name */
    public int f898b = 0;

    /* renamed from: c  reason: collision with root package name */
    public int f899c = Integer.MIN_VALUE;

    /* renamed from: d  reason: collision with root package name */
    public int f900d = Integer.MIN_VALUE;

    /* renamed from: e  reason: collision with root package name */
    public int f901e = 0;

    /* renamed from: f  reason: collision with root package name */
    public int f902f = 0;

    /* renamed from: g  reason: collision with root package name */
    public boolean f903g = false;

    /* renamed from: h  reason: collision with root package name */
    public boolean f904h = false;

    public void a(int i, int i2) {
        this.f899c = i;
        this.f900d = i2;
        this.f904h = true;
        if (this.f903g) {
            if (i2 != Integer.MIN_VALUE) {
                this.f897a = i2;
            }
            if (i != Integer.MIN_VALUE) {
                this.f898b = i;
                return;
            }
            return;
        }
        if (i != Integer.MIN_VALUE) {
            this.f897a = i;
        }
        if (i2 != Integer.MIN_VALUE) {
            this.f898b = i2;
        }
    }
}