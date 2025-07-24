package c.e.b.ef;

import android.content.Context;
import java.util.ArrayList;

/* compiled from: TutorialAdapter.java */
/* loaded from: classes2.dex */
public class g extends b.c0.a.a {

    /* renamed from: b  reason: collision with root package name */
    public Context f4728b;

    /* renamed from: c  reason: collision with root package name */
    public ArrayList<c.e.b.hf.f> f4729c;

    public g(Context context, ArrayList<c.e.b.hf.f> arrayList) {
        this.f4729c = new ArrayList<>();
        this.f4728b = context;
        this.f4729c = arrayList;
    }

    @Override // b.c0.a.a
    public int a() {
        return this.f4729c.size();
    }
}