package g;

import com.google.common.primitives.UnsignedBytes;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.RandomAccess;

/* compiled from: Options.java */
/* loaded from: classes2.dex */
public final class q extends AbstractList<h> implements RandomAccess {

    /* renamed from: b  reason: collision with root package name */
    public final h[] f6200b;

    /* renamed from: c  reason: collision with root package name */
    public final int[] f6201c;

    public q(h[] hVarArr, int[] iArr) {
        this.f6200b = hVarArr;
        this.f6201c = iArr;
    }

    public static void a(long j, e eVar, int i, List<h> list, int i2, int i3, List<Integer> list2) {
        int i4;
        int i5;
        int i6;
        int i7;
        int i8;
        e eVar2;
        if (i2 < i3) {
            for (int i9 = i2; i9 < i3; i9++) {
                if (list.get(i9).l() < i) {
                    throw new AssertionError();
                }
            }
            h hVar = list.get(i2);
            h hVar2 = list.get(i3 - 1);
            if (i == hVar.l()) {
                int intValue = list2.get(i2).intValue();
                int i10 = i2 + 1;
                i5 = i10;
                i4 = intValue;
                hVar = list.get(i10);
            } else {
                i4 = -1;
                i5 = i2;
            }
            long j2 = 4;
            if (hVar.f(i) != hVar2.f(i)) {
                int i11 = 1;
                for (int i12 = i5 + 1; i12 < i3; i12++) {
                    if (list.get(i12 - 1).f(i) != list.get(i12).f(i)) {
                        i11++;
                    }
                }
                long j3 = j + ((int) (eVar.f6176d / 4)) + 2 + (i11 * 2);
                eVar.W(i11);
                eVar.W(i4);
                for (int i13 = i5; i13 < i3; i13++) {
                    byte f2 = list.get(i13).f(i);
                    if (i13 == i5 || f2 != list.get(i13 - 1).f(i)) {
                        eVar.W(f2 & UnsignedBytes.MAX_VALUE);
                    }
                }
                e eVar3 = new e();
                int i14 = i5;
                while (i14 < i3) {
                    byte f3 = list.get(i14).f(i);
                    int i15 = i14 + 1;
                    int i16 = i15;
                    while (true) {
                        if (i16 >= i3) {
                            i7 = i3;
                            break;
                        } else if (f3 != list.get(i16).f(i)) {
                            i7 = i16;
                            break;
                        } else {
                            i16++;
                        }
                    }
                    if (i15 == i7 && i + 1 == list.get(i14).l()) {
                        eVar.W(list2.get(i14).intValue());
                        i8 = i7;
                        eVar2 = eVar3;
                    } else {
                        eVar.W((int) ((((int) (eVar3.f6176d / j2)) + j3) * (-1)));
                        i8 = i7;
                        eVar2 = eVar3;
                        a(j3, eVar3, i + 1, list, i14, i7, list2);
                    }
                    eVar3 = eVar2;
                    i14 = i8;
                    j2 = 4;
                }
                e eVar4 = eVar3;
                eVar.l(eVar4, eVar4.f6176d);
                return;
            }
            int i17 = 0;
            int min = Math.min(hVar.l(), hVar2.l());
            for (int i18 = i; i18 < min && hVar.f(i18) == hVar2.f(i18); i18++) {
                i17++;
            }
            long j4 = 1 + j + ((int) (eVar.f6176d / 4)) + 2 + i17;
            eVar.W(-i17);
            eVar.W(i4);
            int i19 = i;
            while (true) {
                i6 = i + i17;
                if (i19 >= i6) {
                    break;
                }
                eVar.W(hVar.f(i19) & UnsignedBytes.MAX_VALUE);
                i19++;
            }
            if (i5 + 1 == i3) {
                if (i6 == list.get(i5).l()) {
                    eVar.W(list2.get(i5).intValue());
                    return;
                }
                throw new AssertionError();
            }
            e eVar5 = new e();
            eVar.W((int) ((((int) (eVar5.f6176d / 4)) + j4) * (-1)));
            a(j4, eVar5, i6, list, i5, i3, list2);
            eVar.l(eVar5, eVar5.f6176d);
            return;
        }
        throw new AssertionError();
    }

    /* JADX WARN: Code restructure failed: missing block: B:50:0x00c3, code lost:
        continue;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static q b(h... hVarArr) {
        if (hVarArr.length == 0) {
            return new q(new h[0], new int[]{0, -1});
        }
        ArrayList arrayList = new ArrayList(Arrays.asList(hVarArr));
        Collections.sort(arrayList);
        ArrayList arrayList2 = new ArrayList();
        for (int i = 0; i < arrayList.size(); i++) {
            arrayList2.add(-1);
        }
        for (int i2 = 0; i2 < arrayList.size(); i2++) {
            arrayList2.set(Collections.binarySearch(arrayList, hVarArr[i2]), Integer.valueOf(i2));
        }
        if (((h) arrayList.get(0)).l() != 0) {
            int i3 = 0;
            while (i3 < arrayList.size()) {
                h hVar = (h) arrayList.get(i3);
                int i4 = i3 + 1;
                int i5 = i4;
                while (i5 < arrayList.size()) {
                    h hVar2 = (h) arrayList.get(i5);
                    Objects.requireNonNull(hVar2);
                    if (!hVar2.j(0, hVar, 0, hVar.l())) {
                        break;
                    } else if (hVar2.l() != hVar.l()) {
                        if (((Integer) arrayList2.get(i5)).intValue() > ((Integer) arrayList2.get(i3)).intValue()) {
                            arrayList.remove(i5);
                            arrayList2.remove(i5);
                        } else {
                            i5++;
                        }
                    } else {
                        throw new IllegalArgumentException("duplicate option: " + hVar2);
                    }
                }
                i3 = i4;
            }
            e eVar = new e();
            a(0L, eVar, 0, arrayList, 0, arrayList.size(), arrayList2);
            int i6 = (int) (eVar.f6176d / 4);
            int[] iArr = new int[i6];
            for (int i7 = 0; i7 < i6; i7++) {
                iArr[i7] = eVar.readInt();
            }
            if (eVar.f()) {
                return new q((h[]) hVarArr.clone(), iArr);
            }
            throw new AssertionError();
        }
        throw new IllegalArgumentException("the empty byte string is not a supported option");
    }

    @Override // java.util.AbstractList, java.util.List
    public Object get(int i) {
        return this.f6200b[i];
    }

    @Override // java.util.AbstractCollection, java.util.Collection, java.util.List
    public final int size() {
        return this.f6200b.length;
    }
}