package b.d.a.e;

import android.util.Size;
import java.util.Comparator;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class z implements Comparator {

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ z f1355b = new z();

    @Override // java.util.Comparator
    public final int compare(Object obj, Object obj2) {
        Size size = (Size) obj;
        Size size2 = (Size) obj2;
        return Long.signum((size.getWidth() * size.getHeight()) - (size2.getWidth() * size2.getHeight()));
    }
}