/** * @author: AirLab / Field Robotics Center
 *
 * @attention Copyright (C) 2016
 * @attention Carnegie Mellon University
 * @attention All rights reserved
 *
 * @attention LIMITED RIGHTS:
 * @attention The US Government is granted Limited Rights to this Data.
 *            Use, duplication, or disclosure is subject to the
 *            restrictions as stated in Agreement AFS12-1642.
 */
/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Ioan Sucan */

#ifndef OMPL_DATASTRUCTURES_GRID_B_
#define OMPL_DATASTRUCTURES_GRID_B_

#include "ompl/datastructures/GridN.h"
#include "ompl/datastructures/BinaryHeap.h"

namespace ompl
{

    /** \brief This class defines a grid that keeps track of its boundary:
     * it distinguishes between interior and exterior cells.  */
    template < typename _T,
               class LessThanExternal = std::less<_T>,
               class LessThanInternal = LessThanExternal >
    class GridB : public GridN<_T>
    {
    public:

        /// Definition of a cell in this grid
        typedef typename GridN<_T>::Cell      Cell;

        /// The datatype for arrays of cells
        typedef typename GridN<_T>::CellArray CellArray;

        /// Datatype for cell coordinates
        typedef typename GridN<_T>::Coord     Coord;

    protected:

        /// \cond IGNORE
        // the type of cell here needs an extra pointer to allow the updatable heap to work fast
        // however, this stays hidden from the user
        struct CellX : public Cell
        {
            CellX(void) : Cell()
            {
            }

            virtual ~CellX(void)
            {
            }

            void *heapElement;
        };

        /// \endcond

    public:

        /// Event to be called when a cell's priority is to be updated
        typedef void (*EventCellUpdate)(Cell*, void*);

        /// Constructor
        explicit
        GridB(unsigned int dimension) : GridN<_T>(dimension)
        {
            setupHeaps();
        }

        virtual ~GridB(void)
        {
            clearHeaps();
        }

        /// Set the function callback and to be called when a cell's
        /// priority is updated
        void onCellUpdate(EventCellUpdate event, void *arg)
        {
            eventCellUpdate_ = event;
            eventCellUpdateData_ = arg;
        }

        /// Return the cell that is at the top of the heap maintaining internal cells
        Cell* topInternal(void) const
        {
            Cell* top = static_cast<Cell*>(internal_.top()->data);
            return top ? top : topExternal();
        }

        /// Return the cell that is at the top of the heap maintaining external cells
        Cell* topExternal(void) const
        {
            Cell* top = static_cast<Cell*>(external_.top()->data);
            return top ? top : topInternal();
        }

        /// Return the number of internal cells
        unsigned int countInternal(void) const
        {
            return internal_.size();
        }

        /// Return the number of external cells
        unsigned int countExternal(void) const
        {
            return external_.size();
        }

        /// Return the fraction of external cells
        double fracExternal(void) const
        {
            return external_.empty() ? 0.0 : (double)(external_.size()) / (double)(external_.size() + internal_.size());
        }

        /// Return the fraction of internal cells
        double fracInternal(void) const
        {
            return 1.0 - fracExternal();
        }

        /// Update the position in the heaps for a particular cell.
        void update(Cell* cell)
        {
            eventCellUpdate_(cell, eventCellUpdateData_);
            if (cell->border)
                external_.update(reinterpret_cast<typename externalBHeap::Element*>
                                  (static_cast<CellX*>(cell)->heapElement));
            else
                internal_.update(reinterpret_cast<typename internalBHeap::Element*>
                                  (static_cast<CellX*>(cell)->heapElement));
        }

            /// Update all cells and reconstruct the heaps
        void updateAll(void)
        {
            std::vector< Cell* > cells;
            this->getCells(cells);
            for (int i = cells.size() - 1 ; i >= 0 ; --i)
                eventCellUpdate_(cells[i], eventCellUpdateData_);
            external_.rebuild();
            internal_.rebuild();
        }

        /// Create a cell but do not add it to the grid; update neighboring cells however
        virtual Cell* createCell(const Coord& coord, CellArray *nbh = NULL)
        {
            CellX* cell = new CellX();
            cell->coord = coord;

            CellArray *list = nbh ? nbh : new CellArray();
            this->neighbors(cell->coord, *list);

            for (typename CellArray::iterator cl = list->begin() ; cl != list->end() ; ++cl)
            {
                CellX* c = static_cast<CellX*>(*cl);
                bool wasBorder = c->border;
                c->neighbors++;
                if (c->border && c->neighbors >= GridN<_T>::interiorCellNeighborsLimit_)
                    c->border = false;

                eventCellUpdate_(c, eventCellUpdateData_);

                if (c->border)
                    external_.update(reinterpret_cast<typename externalBHeap::Element*>(c->heapElement));
                else
                {
                    if (wasBorder)
                    {
                        external_.remove(reinterpret_cast<typename externalBHeap::Element*>(c->heapElement));
                        internal_.insert(c);
                    }
                    else
                        internal_.update(reinterpret_cast<typename internalBHeap::Element*>(c->heapElement));
                }
            }

            cell->neighbors = GridN<_T>::numberOfBoundaryDimensions(cell->coord) + list->size();
            if (cell->border && cell->neighbors >= GridN<_T>::interiorCellNeighborsLimit_)
                cell->border = false;

            if (!nbh)
                delete list;

            return static_cast<Cell*>(cell);
        }

        /// Add the cell to the grid
        virtual void add(Cell* cell)
        {
            CellX* ccell = static_cast<CellX*>(cell);
            eventCellUpdate_(ccell, eventCellUpdateData_);

            GridN<_T>::add(cell);

            if (cell->border)
                external_.insert(ccell);
            else
                internal_.insert(ccell);
        }

        /// Remove a cell from the grid
        virtual bool remove(Cell* cell)
        {
            if (cell)
            {
                CellArray *list = new CellArray();
                this->neighbors(cell->coord, *list);

                for (typename CellArray::iterator cl = list->begin() ; cl != list->end() ; ++cl)
                {
                    CellX* c = static_cast<CellX*>(*cl);
                    bool wasBorder = c->border;
                    c->neighbors--;
                    if (!c->border && c->neighbors < GridN<_T>::interiorCellNeighborsLimit_)
                        c->border = true;

                    eventCellUpdate_(c, eventCellUpdateData_);

                    if (c->border)
                    {
                        if (wasBorder)
                            external_.update(reinterpret_cast<typename externalBHeap::Element*>(c->heapElement));
                        else
                        {
                            internal_.remove(reinterpret_cast<typename internalBHeap::Element*>(c->heapElement));
                            external_.insert(c);
                        }
                    }
                    else
                        internal_.update(reinterpret_cast<typename internalBHeap::Element*>(c->heapElement));
                }

                delete list;

                typename GridN<_T>::CoordHash::iterator pos = GridN<_T>::hash_.find(&cell->coord);
                if (pos != GridN<_T>::hash_.end())
                {
                    GridN<_T>::hash_.erase(pos);
                    CellX* cx = static_cast<CellX*>(cell);
                    if (cx->border)
                        external_.remove(reinterpret_cast<typename externalBHeap::Element*>(cx->heapElement));
                    else
                        internal_.remove(reinterpret_cast<typename internalBHeap::Element*>(cx->heapElement));
                    return true;
                }
            }
            return false;
        }

        virtual void clear(void)
        {
            GridN<_T>::clear();
            clearHeaps();
        }

        virtual void status(std::ostream &out = std::cout) const
        {
            GridN<_T>::status(out);
            out << countInternal() << " internal cells" << std::endl;
            out << countExternal() << " external cells" << std::endl;
        }

    protected:

        /// Pointer to function to be called when a cell needs to be updated
        EventCellUpdate          eventCellUpdate_;

        /// Data to be passed to function pointer above
        void                    *eventCellUpdateData_;

        /// Default no-op update routine for a cell
        static void noCellUpdate(Cell*, void*)
        {
        }

        /// Set the update procedure for the heaps of internal and external cells
        void setupHeaps(void)
        {
            eventCellUpdate_     = &noCellUpdate;
            eventCellUpdateData_ = NULL;
            internal_.onAfterInsert(&setHeapElementI, NULL);
            external_.onAfterInsert(&setHeapElementE, NULL);
        }

        /// Clear the data from both heaps
        void clearHeaps(void)
        {
            internal_.clear();
            external_.clear();
        }

        /// Define order for internal cells
        struct LessThanInternalCell
        {
            bool operator()(const CellX* const a, const CellX* const b) const
            {
                return lt_(a->data, b->data);
            }

        private:
            LessThanInternal lt_;
        };

        /// Define order for external cells
        struct LessThanExternalCell
        {
            bool operator()(const CellX* const a, const CellX* const b) const
            {
                return lt_(a->data, b->data);
            }
        private:
            LessThanExternal lt_;
        };

        /// Datatype for a heap of cells containing interior cells
        typedef BinaryHeap< CellX*, LessThanInternalCell > internalBHeap;

        /// Datatype for a heap of cells containing exterior cells
        typedef BinaryHeap< CellX*, LessThanExternalCell > externalBHeap;

        /// Routine used internally for keeping track of binary heap elements for internal cells
        static void setHeapElementI(typename internalBHeap::Element* element, void *)
        {
            element->data->heapElement = reinterpret_cast<void*>(element);
        }

        /// Routine used internally for keeping track of binary heap elements for external cells
        static void setHeapElementE(typename externalBHeap::Element* element, void *)
        {
            element->data->heapElement = reinterpret_cast<void*>(element);
        }

        /// The heap of interior cells
        internalBHeap internal_;

        /// The heap of external cells
        externalBHeap external_;
    };

}

#endif
