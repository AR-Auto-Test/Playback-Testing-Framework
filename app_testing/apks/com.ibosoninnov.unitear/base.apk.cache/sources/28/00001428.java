package com.google.android.gms.internal.vision;

import c.b.a.a.a;
import java.util.AbstractList;
import java.util.Arrays;
import java.util.RandomAccess;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzlb<E> extends zzhj<E> implements RandomAccess {
    private static final zzlb<Object> zza;
    private E[] zzb;
    private int zzc;

    static {
        zzlb<Object> zzlbVar = new zzlb<>(new Object[0], 0);
        zza = zzlbVar;
        zzlbVar.zzb();
    }

    public zzlb() {
        this(new Object[10], 0);
    }

    private final void zzb(int i) {
        if (i < 0 || i >= this.zzc) {
            throw new IndexOutOfBoundsException(zzc(i));
        }
    }

    private final String zzc(int i) {
        return a.h(35, "Index:", i, ", Size:", this.zzc);
    }

    public static <E> zzlb<E> zzd() {
        return (zzlb<E>) zza;
    }

    @Override // com.google.android.gms.internal.vision.zzhj, java.util.AbstractList, java.util.AbstractCollection, java.util.Collection, java.util.List
    public final boolean add(E e2) {
        zzc();
        int i = this.zzc;
        E[] eArr = this.zzb;
        if (i == eArr.length) {
            this.zzb = (E[]) Arrays.copyOf(eArr, ((i * 3) / 2) + 1);
        }
        E[] eArr2 = this.zzb;
        int i2 = this.zzc;
        this.zzc = i2 + 1;
        eArr2[i2] = e2;
        ((AbstractList) this).modCount++;
        return true;
    }

    @Override // java.util.AbstractList, java.util.List
    public final E get(int i) {
        zzb(i);
        return this.zzb[i];
    }

    @Override // com.google.android.gms.internal.vision.zzhj, java.util.AbstractList, java.util.List
    public final E remove(int i) {
        int i2;
        zzc();
        zzb(i);
        E[] eArr = this.zzb;
        E e2 = eArr[i];
        if (i < this.zzc - 1) {
            System.arraycopy(eArr, i + 1, eArr, i, (i2 - i) - 1);
        }
        this.zzc--;
        ((AbstractList) this).modCount++;
        return e2;
    }

    @Override // com.google.android.gms.internal.vision.zzhj, java.util.AbstractList, java.util.List
    public final E set(int i, E e2) {
        zzc();
        zzb(i);
        E[] eArr = this.zzb;
        E e3 = eArr[i];
        eArr[i] = e2;
        ((AbstractList) this).modCount++;
        return e3;
    }

    @Override // java.util.AbstractCollection, java.util.Collection, java.util.List
    public final int size() {
        return this.zzc;
    }

    @Override // com.google.android.gms.internal.vision.zzjl
    public final /* synthetic */ zzjl zza(int i) {
        if (i >= this.zzc) {
            return new zzlb(Arrays.copyOf(this.zzb, i), this.zzc);
        }
        throw new IllegalArgumentException();
    }

    private zzlb(E[] eArr, int i) {
        this.zzb = eArr;
        this.zzc = i;
    }

    @Override // com.google.android.gms.internal.vision.zzhj, java.util.AbstractList, java.util.List
    public final void add(int i, E e2) {
        int i2;
        zzc();
        if (i >= 0 && i <= (i2 = this.zzc)) {
            E[] eArr = this.zzb;
            if (i2 < eArr.length) {
                System.arraycopy(eArr, i, eArr, i + 1, i2 - i);
            } else {
                E[] eArr2 = (E[]) new Object[a.m(i2, 3, 2, 1)];
                System.arraycopy(eArr, 0, eArr2, 0, i);
                System.arraycopy(this.zzb, i, eArr2, i + 1, this.zzc - i);
                this.zzb = eArr2;
            }
            this.zzb[i] = e2;
            this.zzc++;
            ((AbstractList) this).modCount++;
            return;
        }
        throw new IndexOutOfBoundsException(zzc(i));
    }
}