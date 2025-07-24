package c.a.a.x.c;

import android.graphics.Path;
import java.util.ArrayList;
import java.util.List;

/* compiled from: MaskKeyframeAnimation.java */
/* loaded from: classes.dex */
public class g {

    /* renamed from: a  reason: collision with root package name */
    public final List<a<c.a.a.z.k.k, Path>> f3237a;

    /* renamed from: b  reason: collision with root package name */
    public final List<a<Integer, Integer>> f3238b;

    /* renamed from: c  reason: collision with root package name */
    public final List<c.a.a.z.k.f> f3239c;

    public g(List<c.a.a.z.k.f> list) {
        this.f3239c = list;
        this.f3237a = new ArrayList(list.size());
        this.f3238b = new ArrayList(list.size());
        for (int i = 0; i < list.size(); i++) {
            this.f3237a.add(list.get(i).f3326b.a());
            this.f3238b.add(list.get(i).f3327c.a());
        }
    }
}