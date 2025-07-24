package c.a.a;

import android.graphics.Rect;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/* compiled from: LottieComposition.java */
/* loaded from: classes.dex */
public class d {

    /* renamed from: c  reason: collision with root package name */
    public Map<String, List<c.a.a.z.l.e>> f3039c;

    /* renamed from: d  reason: collision with root package name */
    public Map<String, k> f3040d;

    /* renamed from: e  reason: collision with root package name */
    public Map<String, c.a.a.z.c> f3041e;

    /* renamed from: f  reason: collision with root package name */
    public List<c.a.a.z.h> f3042f;

    /* renamed from: g  reason: collision with root package name */
    public b.f.i<c.a.a.z.d> f3043g;

    /* renamed from: h  reason: collision with root package name */
    public b.f.e<c.a.a.z.l.e> f3044h;
    public List<c.a.a.z.l.e> i;
    public Rect j;
    public float k;
    public float l;
    public float m;
    public boolean n;

    /* renamed from: a  reason: collision with root package name */
    public final s f3037a = new s();

    /* renamed from: b  reason: collision with root package name */
    public final HashSet<String> f3038b = new HashSet<>();
    public int o = 0;

    public void a(String str) {
        c.a.a.c0.c.b(str);
        this.f3038b.add(str);
    }

    public float b() {
        return (c() / this.m) * 1000.0f;
    }

    public float c() {
        return this.l - this.k;
    }

    /* JADX WARN: Code restructure failed: missing block: B:11:0x003a, code lost:
        if (r3.substring(0, r3.length() - 1).equalsIgnoreCase(r7) != false) goto L11;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public c.a.a.z.h d(String str) {
        this.f3042f.size();
        for (int i = 0; i < this.f3042f.size(); i++) {
            c.a.a.z.h hVar = this.f3042f.get(i);
            boolean z = true;
            if (!hVar.f3281a.equalsIgnoreCase(str)) {
                if (hVar.f3281a.endsWith("\r")) {
                    String str2 = hVar.f3281a;
                }
                z = false;
            }
            if (z) {
                return hVar;
            }
        }
        return null;
    }

    public c.a.a.z.l.e e(long j) {
        return this.f3044h.e(j, null);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder("LottieComposition:\n");
        for (c.a.a.z.l.e eVar : this.i) {
            sb.append(eVar.a("\t"));
        }
        return sb.toString();
    }
}