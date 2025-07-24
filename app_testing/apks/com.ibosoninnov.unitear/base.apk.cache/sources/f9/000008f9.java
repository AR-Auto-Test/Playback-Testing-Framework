package c.d.b.a.p;

import com.google.ar.sceneform.collision.RayHit;
import java.util.Comparator;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class a implements Comparator {

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ a f4312b = new a();

    @Override // java.util.Comparator
    public final int compare(Object obj, Object obj2) {
        return Float.compare(((RayHit) obj).getDistance(), ((RayHit) obj2).getDistance());
    }
}