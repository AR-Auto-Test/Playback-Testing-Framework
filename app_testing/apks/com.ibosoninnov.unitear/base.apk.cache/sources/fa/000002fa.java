package b.d.b.d1;

import java.util.ArrayList;
import java.util.List;

/* compiled from: SurfaceCombination.java */
/* loaded from: classes.dex */
public final class d1 {

    /* renamed from: a  reason: collision with root package name */
    public final List<e1> f1443a = new ArrayList();

    public static void a(List<int[]> list, int i, int[] iArr, int i2) {
        boolean z;
        if (i2 >= iArr.length) {
            list.add((int[]) iArr.clone());
            return;
        }
        for (int i3 = 0; i3 < i; i3++) {
            int i4 = 0;
            while (true) {
                if (i4 >= i2) {
                    z = false;
                    break;
                } else if (i3 == iArr[i4]) {
                    z = true;
                    break;
                } else {
                    i4++;
                }
            }
            if (!z) {
                iArr[i2] = i3;
                a(list, i, iArr, i2 + 1);
            }
        }
    }
}