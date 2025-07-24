package com.google.common.collect;

import com.google.common.annotations.GwtCompatible;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.CompatibleWith;
import java.util.Collection;
import java.util.Iterator;
import java.util.Set;

@GwtCompatible
/* loaded from: classes.dex */
public interface Multiset<E> extends Collection<E> {

    /* loaded from: classes.dex */
    public interface Entry<E> {
        boolean equals(Object obj);

        int getCount();

        E getElement();

        int hashCode();

        String toString();
    }

    @CanIgnoreReturnValue
    int add(E e2, int i);

    @Override // java.util.Collection, com.google.common.collect.Multiset
    @CanIgnoreReturnValue
    boolean add(E e2);

    @Override // java.util.Collection, com.google.common.collect.Multiset
    boolean contains(Object obj);

    @Override // java.util.Collection
    boolean containsAll(Collection<?> collection);

    int count(@CompatibleWith("E") Object obj);

    Set<E> elementSet();

    Set<Entry<E>> entrySet();

    @Override // com.google.common.collect.Multiset
    boolean equals(Object obj);

    @Override // com.google.common.collect.Multiset
    int hashCode();

    @Override // java.util.Collection, java.lang.Iterable, com.google.common.collect.Multiset
    Iterator<E> iterator();

    @CanIgnoreReturnValue
    int remove(@CompatibleWith("E") Object obj, int i);

    @Override // java.util.Collection, com.google.common.collect.Multiset
    @CanIgnoreReturnValue
    boolean remove(Object obj);

    @Override // java.util.Collection, com.google.common.collect.Multiset
    @CanIgnoreReturnValue
    boolean removeAll(Collection<?> collection);

    @Override // java.util.Collection, com.google.common.collect.Multiset
    @CanIgnoreReturnValue
    boolean retainAll(Collection<?> collection);

    @CanIgnoreReturnValue
    int setCount(E e2, int i);

    @CanIgnoreReturnValue
    boolean setCount(E e2, int i, int i2);

    @Override // java.util.Collection, com.google.common.collect.Multiset
    int size();

    String toString();
}