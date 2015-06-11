//
//  sudoku.hpp
//
//  Created by Jordi Pont on 2/1/15.
//  Copyright (c) 2015 Jordi Pont. All rights reserved.
//

#ifndef Sudoku_sudoku_hpp
#define Sudoku_sudoku_hpp

/* Class that contains and solves the Sudoku of N*N*N grid 'spaces', i.e., the usual Sudoku is N=3 */
template<std::size_t N>
class Sudoku
{
public:
    /* Default constructor, which:
     *  - Allocates the grid (N*N x N*N, e.g. 9x9)
     *  - Defines the N*N constraints of each type (e.g. 9 columns, etc.)
     */
    Sudoku() : grid(N*N,std::vector<std::unordered_set<char>>(N*N))
    {
        for(std::size_t ii=0; ii<N*N; ++ii)
        {
            constraints.push_back(ColConstraint(ii));
            constraints.push_back(RowConstraint(ii));
            constraints.push_back(BlockConstraint(ii));
        }
    }
    
    /* Function to solve the Sudoku, given the current imposed numbers */
    bool solve()
    {
        std::size_t max_iterations = 1000;
        std::size_t n_iters = 0;
        while(not is_solved())
        {
            /* We impose that cosntrain, removing solved digits
             * from the rest of locations in the constraint */
            for(auto constr:constraints)
                constr.scan_and_impose(grid);
            n_iters++;
            if(n_iters>max_iterations)
                return false;
        }
        
        return true;
    }
    
    /* Sets the value in a particular position in the grid */
    void set_value(std::size_t xpos, std::size_t ypos, char value)
    {
        if (value==0)
            for(char kk=0; kk<N*N; ++kk)
                grid[xpos][ypos].insert(kk+1);
        else
            grid[xpos][ypos].insert(value);
    }
    
    /* Gets the value in a particular position in the grid */
    char get_value(std::size_t xpos, std::size_t ypos)
    {
        if(grid[xpos][ypos].size()==1)
            return *grid[xpos][ypos].begin();
        else
            return 0;
    }
    
    /* Reads a Sudoku from a file, where '0' refers to an empty slot */
    void read(const std::string& filename)
    {
        std::ifstream file(filename);
        if(not file.is_open())
            std::cerr << "Cannot read file: " + filename << std::endl;
        else
        {
            for(std::size_t ii=0; ii<grid.size(); ++ii)
                for(std::size_t jj=0; jj<grid[ii].size(); ++jj)
                {
                    char tmp;
                    file >> tmp;
                    if (tmp=='0')
                        for(char kk=0; kk<N*N; ++kk)
                            grid[ii][jj].insert(kk+1);
                    else
                        grid[ii][jj].insert(tmp-'0');
                }
            file.close();
        }
    }
    
    
    /* Prints the Sudoku to the console, with empty slots represented by '-' */
    void print() const
    {
        std::cout << " -----------------------" << std::endl;
        for(std::size_t ii=0; ii<grid.size(); ++ii)
        {
            std::cout << "| ";
            for(std::size_t jj=0; jj<grid[ii].size(); ++jj)
            {
                if (grid[ii][jj].size()==1)
                    std::cout << (int)*(grid[ii][jj].begin()) << " ";
                else
                    std::cout << "- ";
                if (jj%N == N-1)
                    std::cout << "| ";
            }
            std::cout << std::endl;
            if (ii%N == N-1)
                std::cout << " -----------------------" << std::endl;
        }
        
    }
    
    /* Function to check whether the Sudoku is solved, i.e., whether there is only one option left at all positions */
    bool is_solved()
    {
        bool is_sol = true;
        for(std::size_t ii=0; ii<N*N; ++ii)
            for(std::size_t jj=0; jj<N*N; ++jj)
                if(grid[ii][jj].size()!=1)
                {
                    is_sol = false;
                    ii = N*N;
                    jj = N*N;
                }
        return is_sol;
    }
    
private:
    
    /* Generic class that implements a constraint, i.e.,
     *  a set of positions where there cannot be any repeated number.
     * We will implement the generic features of a constraint (scan all positions and impose no repetitions),
     *  and then we will specialize this class to the three types of constraints in a Sudoku (row, column, and block).
     * This way we don't have to duplicate code (Object-Oriented programming :)
     */
    struct Constraint
    {
        /* Default constructor */
        Constraint(): row_coords(N*N), col_coords(N*N)
        {
        }
        
        /* String identifying the constraint, for debugging purposes */
        std::string id;
        
        /* Coordinates of the constraint, that is, the positions where the constraints must be fulfilled */
        std::vector<std::size_t> row_coords;
        std::vector<std::size_t> col_coords;
        
        /* Scan the constraint, find those positions with only one possible value,
         *  and erase the value from the rest of sets, where it cannot be.
         *  Example: If we find a place where '9' is the only possible number, we erase '9' from the
         *           list of possibilities in the other locations */
        bool scan_and_impose(std::vector<std::vector<std::unordered_set<char>>>& contain)
        {
            bool now_solved = false;
            
            /* All possible numbers */
            std::unordered_set<char> possible;
            for(char ii=0; ii<N*N; ++ii)
                possible.insert(ii+1);
            
            /* Find those that are already solved */
            std::unordered_set<char> solved;
            for(std::size_t ii=0; ii<N*N; ++ii)
                if (contain[row_coords[ii]][col_coords[ii]].size()==1)
                    solved.insert(*(contain[row_coords[ii]][col_coords[ii]].begin()));
            
            /* Erase the solved ones from the rest of positions */
            for(std::size_t ii=0; ii<N*N; ++ii)
                if (contain[row_coords[ii]][col_coords[ii]].size()>1)
                {
                    for(auto er:solved)
                        contain[row_coords[ii]][col_coords[ii]].erase(er);
                    if(contain[row_coords[ii]][col_coords[ii]].size()==1)
                        now_solved = true;
                    
                }
            
            return now_solved;
        }
    };
    
    /* Specialized class for row constraints */
    struct RowConstraint: public Constraint
    {
        /* Constructor receiving the particular row */
        RowConstraint(std::size_t row)
        {
            /* Fill in all the coordinates of the row */
            for(std::size_t ii=0; ii<N*N; ++ii)
            {
                this->row_coords[ii] = row;
                this->col_coords[ii] = ii;
            }
            
            /* Set the identifier string */
            std::stringstream ss;
            ss << row;
            this->id = "Row "+ss.str();
        }
        
    };
    
    /* Specialized class for column constraints */
    struct ColConstraint: public Constraint
    {
        /* Constructor receiving the particular column */
        ColConstraint(std::size_t col)
        {
            /* Fill in all the coordinates of the column */
            for(std::size_t ii=0; ii<N*N; ++ii)
            {
                this->col_coords[ii] = col;
                this->row_coords[ii] = ii;
            }
            
            /* Set the identifier string */
            std::stringstream ss;
            ss << col;
            this->id = "Col "+ss.str();
        }
    };
    
    /* Specialized class for block constraints */
    struct BlockConstraint: public Constraint
    {
        /* Constructor receiving the particular block */
        BlockConstraint(std::size_t block)
        {
            /* Fill in all the coordinates of the block */
            std::size_t block_x = block%3;
            std::size_t block_y = std::floor(block/3);
            std::size_t coord = 0;
            for(std::size_t ii=0; ii<N; ++ii)
            {
                for(std::size_t jj=0; jj<N; ++jj)
                {
                    this->col_coords[coord] = ii+block_x*N;
                    this->row_coords[coord] = jj+block_y*N;
                    coord++;
                }
            }
            
            /* Set the identifier string */
            std::stringstream ss;
            ss << block;
            this->id = "Block "+ss.str();
        }
    };
    
    
    /* Set of constraints container */
    std::vector<Constraint> constraints;
    
    /* Container of all possible numbers at each grid position */
    std::vector<std::vector<std::unordered_set<char>>> grid;
};

#endif
